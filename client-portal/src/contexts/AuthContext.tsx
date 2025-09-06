import React, { createContext, useContext, useEffect, useState } from 'react'
import { api } from '../services/api'

type AuthContextType = {
  isAuthenticated: boolean
  loading: boolean
  user?: { name: string, email: string } | null
  login: () => void
  logout: () => void
}

const AuthContext = createContext<AuthContextType>({
  isAuthenticated: false,
  loading: true,
  user: null,
  login: () => {},
  logout: () => {}
})

export const AuthProvider: React.FC<{children: React.ReactNode}> = ({ children }) => {
  const [loading, setLoading] = useState(true)
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [user, setUser] = useState<AuthContextType['user']>(null)

  useEffect(() => {
    api.getAuthConfig().then(() => {
      setIsAuthenticated(true)
      setUser({ name: 'Demo User', email: 'demo@towerco.local' })
    }).catch(() => {
      setIsAuthenticated(false)
      setUser(null)
    }).finally(() => setLoading(false))
  }, [])

  const login = async () => {
    try {
      const response = await api.login()
      if (response.login_url) {
        window.location.href = response.login_url
      }
    } catch (error) {
      console.error('Login failed:', error)
    }
  }
  
  const logout = async () => {
    try {
      await api.logout()
      setIsAuthenticated(false)
      setUser(null)
      window.location.href = '/login'
    } catch (error) {
      console.error('Logout failed:', error)
    }
  }

  return (
    <AuthContext.Provider value={{ isAuthenticated, loading, user, login, logout }}>
      {children}
    </AuthContext.Provider>
  )
}

export const useAuth = () => useContext(AuthContext)
