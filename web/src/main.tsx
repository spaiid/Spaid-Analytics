import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

// Order matters: tokens define the custom properties everything else reads.
import './styles/tokens.css';
import './styles/base.css';
import './styles/components.css';
import './styles/charts.css';

import { App } from './App';

const container = document.getElementById('root');
if (!container) {
  throw new Error('Missing #root element in index.html');
}

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
