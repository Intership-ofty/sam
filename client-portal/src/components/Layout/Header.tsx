import React from 'react'
import { useAuth } from '../../contexts/AuthContext'
import { useTheme } from '../../contexts/ThemeContext'
import { useTenant } from '../../contexts/TenantContext'
const Header: React.FC = () => {
  const { user, logout } = useAuth()
  const { theme, toggle } = useTheme()
  const { tenantId, setTenantId } = useTenant()
  return (
    <header className="header">
      <strong>Client Portal</strong>
      <div style={{ marginLeft: 'auto', display:'flex', gap:12, alignItems:'center' }}>
        <select value={tenantId} onChange={(e)=>setTenantId(e.target.value)}>
          <option value="default">Default Tenant</option>
          <option value="mno-a">MNO-A</option>
          <option value="mno-b">MNO-B</option>
        </select>
        <button onClick={toggle}>Theme: {theme}</button>
        <span>{user?.name}</span>
        <button onClick={logout}>Logout</button>
      </div>
    </header>
  )
}
export default Header
