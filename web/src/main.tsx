import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './styles/globals.css';
import { App } from './App';
import { applyStoredTheme } from './utils/theme';

// Apply the saved theme before first paint so there is no palette flash.
applyStoredTheme();

const root = createRoot(document.getElementById('root')!);

if (import.meta.env.DEV) {
  root.render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
} else {
  root.render(<App />);
}
