import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import { Amplify } from 'aws-amplify'
import App from './App.jsx'
import awsConfig from './auth/aws-exports.js'

Amplify.configure(awsConfig)

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
