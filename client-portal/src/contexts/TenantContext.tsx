import React, { createContext, useContext, useState } from 'react'

type TenantContextType = {
  tenantId: string
  setTenantId: (id: string) => void
}

const TenantContext = createContext<TenantContextType>({
  tenantId: 'default',
  setTenantId: () => {}
})

export const TenantProvider: React.FC<{children: React.ReactNode}> = ({ children }) => {
  const [tenantId, setTenantId] = useState('default')
  return (
    <TenantContext.Provider value={{ tenantId, setTenantId }}>
      {children}
    </TenantContext.Provider>
  )
}

export const useTenant = () => useContext(TenantContext)
