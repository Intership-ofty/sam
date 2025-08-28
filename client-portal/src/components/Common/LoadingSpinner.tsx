import React from 'react'
const LoadingSpinner: React.FC<{label?: string}> = ({ label }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
    <div style={{ width: 16, height: 16, border: '2px solid #60a5fa', borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />
    {label && <span>{label}</span>}
    <style>{`@keyframes spin{to{transform: rotate(360deg)}}`}</style>
  </div>
)
export default LoadingSpinner
