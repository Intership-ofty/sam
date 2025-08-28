import React from 'react'
import { NavLink } from 'react-router-dom'
const Sidebar: React.FC = () => {
  const link = ({ isActive }: {isActive: boolean}) => (isActive ? 'active' : undefined)
  return (
    <aside className="sidebar">
      <nav>
        <ul>
          <li><NavLink to="/" className={link}>Dashboard</NavLink></li>
          <li><NavLink to="/sites" className={link}>Sites</NavLink></li>
          <li><NavLink to="/kpis" className={link}>KPIs</NavLink></li>
          <li><NavLink to="/alerts" className={link}>Alerts</NavLink></li>
          <li><NavLink to="/sla" className={link}>SLA</NavLink></li>
          <li><NavLink to="/reports" className={link}>Reports</NavLink></li>
          <li><NavLink to="/maintenance" className={link}>Maintenance</NavLink></li>
          <li><NavLink to="/settings" className={link}>Settings</NavLink></li>
        </ul>
      </nav>
    </aside>
  )
}
export default Sidebar
