import { createRoot } from 'react-dom/client';
import { StockDetailView } from './src/views/StockDetailView';

const el = document.getElementById('root');
if (el) createRoot(el).render(<StockDetailView ticker="AAPL" />);
