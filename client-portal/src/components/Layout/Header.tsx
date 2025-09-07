import React from 'react'
import { useAuth } from '../../contexts/AuthContext'
import { useTheme } from '../../contexts/ThemeContext'
import { useTenant } from '../../contexts/TenantContext'

const Header: React.FC = () => {
  const { user, logout, login } = useAuth()
  const { theme, toggle } = useTheme()
  const { tenantId, setTenantId } = useTenant()

  return (
    <header className="header">
      <div className="flex items-center space-x-4">
        <h1 className="text-xl font-bold text-gray-900">Client Portal</h1>
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
          <span className="text-sm text-gray-600">System Online</span>
        </div>
      </div>
      
      <div className="flex items-center space-x-4">
        <select 
          value={tenantId} 
          onChange={(e) => setTenantId(e.target.value)}
          className="px-3 py-1 text-sm border border-gray-300 rounded-md bg-white"
        >
          <option value="default">Default Tenant</option>
          <option value="mno-a">MNO-A</option>
          <option value="mno-b">MNO-B</option>
        </select>
        
        <button 
          onClick={toggle}
          className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded-md transition-colors"
        >
          Theme: {theme}
        </button>
        
        {user ? (
          <div className="flex items-center space-x-3">
            <span className="text-sm text-gray-700 font-medium">
              {user.name}
            </span>
            <button 
              onClick={logout}
              className="btn-secondary text-sm px-3 py-1"
            >
              Logout
            </button>
          </div>
        ) : (
          <button 
            onClick={login}
            className="btn-primary text-sm px-4 py-2"
          >
            Login
          </button>
        )}
      </div>
    </header>
  )
}

export default Header
