import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './styles/globals.css';
import { App } from './App';

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
