import { useRef, useState, useCallback, useEffect, useImperativeHandle, forwardRef } from 'react';
import { CommandPalette } from './CommandPalette';
import { transcribeAudio } from '../api';
import type { PendingAttachment } from '../types';

const SUPPORTED_MIMES = new Set([
  'image/png', 'image/jpeg', 'image/gif', 'image/webp', 'application/pdf',
]);

const MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024; // 20MB

export interface InputAreaHandle {
  /** Attach files dropped elsewhere (e.g. the chat area). */
  addFiles: (files: FileList | File[]) => void;
}

interface InputAreaProps {
  onSend: (text: string, attachments?: PendingAttachment[]) => void;
  onAbort: () => void;
  isRunning: boolean;
  disabled: boolean;
}

export const InputArea = forwardRef<InputAreaHandle, InputAreaProps>(function InputArea(
  { onSend, onAbort, isRunning, disabled }: InputAreaProps,
  ref,
) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [attachments, setAttachments] = useState<PendingAttachment[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);
  // dragenter/dragleave fire for every child too; count depth so the highlight
  // only clears when the pointer actually leaves the drop zone.
  const dragDepthRef = useRef(0);
  const [showCommands, setShowCommands] = useState(false);
  const [cmdFilter, setCmdFilter] = useState('');
  // Feedback for files that couldn't be attached (wrong type / too big / unreadable).
  const [attachNotice, setAttachNotice] = useState('');
  // Sent messages for ↑/↓ recall; histIdx -1 = live draft.
  const historyRef = useRef<string[]>([]);
  const histIdxRef = useRef(-1);
  // Voice input (MediaRecorder → server STT)
  const [recState, setRecState] = useState<'idle' | 'recording' | 'transcribing'>('idle');
  const recorderRef = useRef<MediaRecorder | null>(null);
  const recChunksRef = useRef<Blob[]>([]);

  const stopRecording = useCallback(() => {
    recorderRef.current?.stop();
  }, []);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      recChunksRef.current = [];
      recorder.ondataavailable = (e) => { if (e.data.size > 0) recChunksRef.current.push(e.data); };
      recorder.onstop = async () => {
        stream.getTracks().forEach(t => t.stop());
        setRecState('transcribing');
        try {
          const blob = new Blob(recChunksRef.current, { type: recorder.mimeType });
          const buf = await blob.arrayBuffer();
          let bin = '';
          const bytes = new Uint8Array(buf);
          for (let i = 0; i < bytes.length; i += 0x8000) {
            bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
          }
          const res = await transcribeAudio(btoa(bin), recorder.mimeType);
          if (res.ok && res.text) {
            const el = textareaRef.current;
            if (el) {
              // Route through setValue so the box auto-grows and the caret lands
              // at the end (a raw el.value write skips both).
              setValue(el.value ? `${el.value} ${res.text}` : res.text);
              el.focus();
            }
          } else {
            setAttachNotice(res.error || 'Transcription failed.');
          }
        } catch {
          setAttachNotice('Transcription failed.');
        } finally {
          setRecState('idle');
        }
      };
      recorder.start();
      recorderRef.current = recorder;
      setRecState('recording');
    } catch {
      setAttachNotice('Microphone unavailable (permission denied?).');
    }
  }, []);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (isRunning && textarea && textarea === document.activeElement) {
      textarea.blur();
    }
  }, [isRunning]);

  useEffect(() => {
    if (!attachNotice) return;
    const t = setTimeout(() => setAttachNotice(''), 5000);
    return () => clearTimeout(t);
  }, [attachNotice]);


  const processFiles = useCallback((files: FileList | File[]) => {
    const rejected: string[] = [];
    for (const file of Array.from(files)) {
      if (!SUPPORTED_MIMES.has(file.type)) {
        rejected.push(`${file.name}: unsupported type`);
        continue;
      }
      if (file.size > MAX_ATTACHMENT_BYTES) {
        rejected.push(`${file.name}: over 20MB`);
        continue;
      }
      const reader = new FileReader();
      reader.onload = () => {
        const isImage = file.type.startsWith('image/');
        setAttachments(prev => [...prev, {
          id: crypto.randomUUID(),
          name: file.name,
          mimeType: file.type,
          size: file.size,
          dataUrl: reader.result as string,
          preview: isImage ? URL.createObjectURL(file) : undefined,
        }]);
      };
      reader.onerror = () => setAttachNotice(`Could not read ${file.name}`);
      reader.readAsDataURL(file);
    }
    setAttachNotice(rejected.join(' · '));
  }, []);

  useImperativeHandle(ref, () => ({ addFiles: processFiles }), [processFiles]);

  const removeAttachment = useCallback((id: string) => {
    setAttachments(prev => {
      const att = prev.find(a => a.id === id);
      if (att?.preview) URL.revokeObjectURL(att.preview);
      return prev.filter(a => a.id !== id);
    });
  }, []);

  const setValue = (v: string) => {
    const el = textareaRef.current;
    if (!el) return;
    el.value = v;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 160) + 'px';
    el.selectionStart = el.selectionEnd = v.length;
  };

  const handleSubmit = () => {
    const text = textareaRef.current?.value.trim() ?? '';
    if (!text && attachments.length === 0) return;
    if (text && historyRef.current[historyRef.current.length - 1] !== text) {
      historyRef.current.push(text);
    }
    histIdxRef.current = -1;
    onSend(text, attachments.length > 0 ? attachments : undefined);
    // Revoke all object URLs
    attachments.forEach(a => { if (a.preview) URL.revokeObjectURL(a.preview); });
    setAttachments([]);
    setAttachNotice('');
    if (textareaRef.current) {
      textareaRef.current.value = '';
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // When command palette is open, let it handle Enter/Arrow/Escape
    if (showCommands) {
      if (e.key === 'Enter' || e.key === 'ArrowUp' || e.key === 'ArrowDown') {
        e.preventDefault();
        return;
      }
      if (e.key === 'Escape') {
        e.preventDefault();
        setShowCommands(false);
        return;
      }
    }
    // ↑/↓ recall only when the caret is at the very start, so multi-line editing keeps the arrows.
    const el = textareaRef.current;
    const hist = historyRef.current;
    if (el && hist.length > 0 && !isRunning && el.selectionStart === 0 && el.selectionEnd === 0) {
      if (e.key === 'ArrowUp') {
        const next = histIdxRef.current === -1 ? hist.length - 1 : Math.max(0, histIdxRef.current - 1);
        histIdxRef.current = next;
        e.preventDefault();
        setValue(hist[next]);
        el.selectionStart = el.selectionEnd = 0; // keep caret at start so ↑ keeps cycling
        return;
      }
      if (e.key === 'ArrowDown' && histIdxRef.current !== -1) {
        e.preventDefault();
        if (histIdxRef.current >= hist.length - 1) {
          histIdxRef.current = -1;
          setValue('');
        } else {
          histIdxRef.current += 1;
          setValue(hist[histIdxRef.current]);
          el.selectionStart = el.selectionEnd = 0;
        }
        return;
      }
    }
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      if (!isRunning) handleSubmit();
    }
    if (e.key === 'Escape' && isRunning) {
      e.preventDefault();
      onAbort();
    }
  };

  const handleInput = () => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = 'auto';
      el.style.height = Math.min(el.scrollHeight, 160) + 'px';

      const val = el.value;
      if (val.startsWith('/')) {
        setShowCommands(true);
        setCmdFilter(val.slice(1));
      } else {
        setShowCommands(false);
      }
    }
  };

  const handleCommandSelect = (command: string) => {
    if (textareaRef.current) {
      textareaRef.current.value = command + ' ';
      textareaRef.current.focus();
    }
    setShowCommands(false);
  };

  const handlePaste = (e: React.ClipboardEvent) => {
    const files = e.clipboardData?.files;
    if (files && files.length > 0) {
      e.preventDefault();
      processFiles(files);
    }
  };

  // Only react to file drags, not text selections or element drags.
  const isFileDrag = (e: React.DragEvent) =>
    Array.from(e.dataTransfer?.types ?? []).includes('Files');

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    dragDepthRef.current = 0;
    setIsDragOver(false);
    if (e.dataTransfer?.files.length) {
      processFiles(e.dataTransfer.files);
    }
  };

  const handleDragEnter = (e: React.DragEvent) => {
    if (!isFileDrag(e)) return;
    e.preventDefault();
    dragDepthRef.current += 1;
    setIsDragOver(true);
  };

  const handleDragOver = (e: React.DragEvent) => {
    if (!isFileDrag(e)) return;
    e.preventDefault();  // required so the drop event fires
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);
    if (dragDepthRef.current === 0) setIsDragOver(false);
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes}B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)}KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
  };

  return (
    <div
      style={{
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        padding: '0 20px 20px',
        pointerEvents: 'none',
        zIndex: 10,
      }}
    >
      <div
        className="glass composer-shell"
        onDrop={handleDrop}
        onDragEnter={handleDragEnter}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        style={{
          maxWidth: 768,
          margin: '0 auto',
          borderRadius: 'var(--radius-xl)',
          border: isDragOver ? '2px solid var(--accent)' : '1px solid var(--border)',
          boxShadow: 'var(--shadow-lg)',
          pointerEvents: 'auto',
          overflow: showCommands ? 'visible' : 'hidden',
          position: 'relative',
        }}>
        {/* Rejected-file feedback */}
        {attachNotice && (
          <div style={{
            padding: '8px 16px 0',
            fontSize: 12,
            color: 'var(--danger)',
            wordBreak: 'break-word',
          }}>
            {attachNotice}
          </div>
        )}

        {/* Attachment previews */}
        {attachments.length > 0 && (
          <div style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: 8,
            padding: '10px 16px 0',
          }}>
            {attachments.map(att => (
              <div key={att.id} style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                padding: '4px 8px',
                background: 'var(--bg-secondary)',
                borderRadius: 'var(--radius-md)',
                fontSize: 12,
                color: 'var(--text-secondary)',
                maxWidth: 200,
              }}>
                {att.preview ? (
                  <img
                    src={att.preview}
                    alt={att.name}
                    style={{
                      width: 24, height: 24, borderRadius: 4, objectFit: 'cover',
                    }}
                  />
                ) : (
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M7 3h8l4 4v14H7z" />
                    <path d="M14 3v5h5" />
                  </svg>
                )}
                <span style={{
                  overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1,
                }}>
                  {att.name}
                </span>
                <span style={{ color: 'var(--text-muted)', flexShrink: 0 }}>
                  {formatSize(att.size)}
                </span>
                <button
                  type="button"
                  onClick={() => removeAttachment(att.id)}
                  style={{
                    background: 'none', border: 'none', color: 'var(--text-muted)',
                    cursor: 'pointer', padding: '0 2px', fontSize: 14, lineHeight: 1,
                    flexShrink: 0,
                  }}
                  title="Remove"
                >
                  {'\u00D7'}
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Input row */}
        <div style={{
          display: 'flex',
          alignItems: 'flex-end',
          gap: 0,
          padding: '8px 8px 8px 16px',
          position: 'relative',
        }}>
          {showCommands && (
            <CommandPalette
              filter={cmdFilter}
              onSelect={handleCommandSelect}
              onClose={() => setShowCommands(false)}
            />
          )}
          {/* Attach button */}
          {!isRunning && (
            <>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept="image/png,image/jpeg,image/gif,image/webp,application/pdf"
                style={{ display: 'none' }}
                onChange={(e) => {
                  if (e.target.files) processFiles(e.target.files);
                  e.target.value = '';
                }}
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                disabled={disabled}
                title="Attach file"
                aria-label="Attach file"
                style={{
                  width: 36, height: 36, borderRadius: '50%',
                  background: 'none', border: 'none',
                  color: disabled ? 'var(--text-muted)' : 'var(--text-secondary)',
                  cursor: disabled ? 'default' : 'pointer',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  flexShrink: 0, marginBottom: 4,
                }}
              >
                <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M19 11l-8 8a4 4 0 0 1-6-6l9-9a2.7 2.7 0 0 1 4 4l-8.5 8.5a1.4 1.4 0 0 1-2-2l7.5-7.5" />
                </svg>
              </button>
              <button
                type="button"
                onClick={() => (recState === 'recording' ? stopRecording() : startRecording())}
                disabled={disabled || recState === 'transcribing'}
                title={recState === 'recording' ? 'Stop recording' : recState === 'transcribing' ? 'Transcribing…' : 'Voice input'}
                aria-label={recState === 'recording' ? 'Stop recording' : 'Voice input'}
                style={{
                  width: 36, height: 36, borderRadius: '50%',
                  background: recState === 'recording' ? 'var(--danger)' : 'none',
                  border: 'none',
                  color: recState === 'recording'
                    ? 'white'
                    : disabled || recState === 'transcribing'
                      ? 'var(--text-muted)'
                      : 'var(--text-secondary)',
                  cursor: disabled ? 'default' : 'pointer',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  flexShrink: 0, marginBottom: 4,
                }}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="9" y="2" width="6" height="11" rx="3" />
                  <path d="M5 10v1a7 7 0 0 0 14 0v-1" />
                  <path d="M12 18v4" />
                </svg>
              </button>
            </>
          )}

          <textarea
            ref={textareaRef}
            rows={1}
            placeholder={
              isDragOver
                ? 'Drop files here...'
                : disabled
                  ? 'Connecting...'
                  : isRunning
                    ? 'Run in progress. Stop to send the next message.'
                    : 'Message RUNE...'
            }
            disabled={disabled}
            readOnly={isRunning}
            aria-disabled={disabled || isRunning}
            autoFocus
            onKeyDown={handleKeyDown}
            onInput={handleInput}
            onPaste={handlePaste}
            style={{
              flex: 1,
              background: 'transparent',
              border: 'none',
              padding: '10px 8px',
              color: 'var(--text-primary)',
              fontSize: 15,
              lineHeight: 1.5,
              minHeight: 44,
              maxHeight: 160,
              outline: 'none',
              resize: 'none',
              fontFamily: 'var(--font-sans)',
              opacity: isRunning ? 0.7 : 1,
              cursor: isRunning ? 'not-allowed' : 'text',
              pointerEvents: isRunning ? 'none' : 'auto',
            }}
          />
          {isRunning ? (
            <button
              type="button"
              onClick={onAbort}
              title="Stop"
              aria-label="Stop generation"
              style={{
                width: 36, height: 36, borderRadius: '50%',
                background: 'var(--danger)', color: 'white',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                flexShrink: 0, fontSize: 14, marginBottom: 4,
              }}
            >
              {'\u25A0'}
            </button>
          ) : (
            <button
              type="button"
              onClick={handleSubmit}
              disabled={disabled}
              title="Send"
              aria-label="Send message"
              style={{
                width: 36, height: 36, borderRadius: '50%',
                background: disabled ? 'var(--bg-tertiary)' : 'var(--accent)',
                color: disabled ? 'var(--text-muted)' : '#0A1319',
                fontWeight: 700,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                flexShrink: 0, fontSize: 16, marginBottom: 4,
              }}
            >
              {'\u2191'}
            </button>
          )}
        </div>
        <div style={{
          fontSize: 11,
          color: 'var(--text-muted)',
          padding: '0 16px 8px',
          opacity: 0.7,
        }}>
          {isRunning
            ? 'Current run active · Stop to send a new message'
            : disabled
              ? ''
              : isDragOver
                ? 'Drop to attach'
                : 'Enter to send \u00B7 Shift+Enter for newline \u00B7 \u2191 recall'}
        </div>
      </div>
    </div>
  );
});
