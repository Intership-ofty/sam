import React from 'react'
import { NavLink } from 'react-router-dom'

const Sidebar: React.FC = () => {
  const navItems = [
    { path: '/', label: 'Dashboard', icon: '📊' },
    { path: '/sites', label: 'Sites', icon: '🏗️' },
    { path: '/kpis', label: 'KPIs', icon: '📈' },
    { path: '/kpis/management', label: 'Gestion KPIs', icon: '⚙️' },
    { path: '/alerts', label: 'Alerts', icon: '⚠️' },
    { path: '/reports', label: 'Reports', icon: '📋' },
    { path: '/sla', label: 'SLA', icon: '🎯' },
    { path: '/maintenance', label: 'Maintenance', icon: '🔧' },
    { path: '/settings', label: 'Settings', icon: '⚙️' }
  ]

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h2 className="text-xl font-bold text-white">Towerco AIOps</h2>
        <p className="text-sm text-gray-400 mt-1">Client Portal</p>
      </div>
      <nav className="sidebar-nav">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) => 
              `nav-item ${isActive ? 'active' : ''}`
            }
          >
            <span className="nav-item-icon">{item.icon}</span>
            {item.label}
          </NavLink>
        ))}
      </nav>
    </aside>
  )
}

export default Sidebar
