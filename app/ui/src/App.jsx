import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import './App.css';
import Classifier from './pages/classifier/classifier';
import Generator from './pages/generator/Generator';

function App() {
  return (
    <Router>
      <div className="app-root">
        
        {/* Top Navbar */}
        <nav className="navbar">
          <h1>MRI AI Tool</h1>
          <div className="nav-links">
            <Link to="/">Classifier</Link>
            <Link to="/generator">Generator</Link>
          </div>
        </nav>

        {/* Page Content */}
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Classifier />} />
            <Route path="/generator" element={<Generator />} />
          </Routes>
        </main>

        {/* Disclaimer */}
        <footer className="footer">
          <p>Don't make medical decisions based on this app</p>
        </footer>
      </div>
    </Router>
  );
}

export default App;
