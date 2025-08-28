import React from 'react'
import { useAuth } from '../../contexts/AuthContext'
export default function LoginPage() {
  const { login } = useAuth()
  return (
    <div className="center">
      <div className="card">
        <h2>Login</h2>
        <p>You're behind OAuth2-Proxy. Click the button to start SSO.</p>
        <button onClick={login}>Login with Keycloak</button>
      </div>
    </div>
  )
}
