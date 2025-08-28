import React, { createContext, useContext, useEffect, useState } from 'react'
type Theme = 'dark' | 'light'
const ThemeContext = createContext<{ theme: Theme, toggle: () => void }>({
  theme: 'dark', toggle: () => {}
})
export const ThemeProvider: React.FC<{children: React.ReactNode}> = ({ children }) => {
  const [theme, setTheme] = useState<Theme>('dark')
  useEffect(() => { document.documentElement.dataset.theme = theme }, [theme])
  return (
    <ThemeContext.Provider value={{ theme, toggle: () => setTheme(t => t === 'dark' ? 'light' : 'dark') }}>
      {children}
    </ThemeContext.Provider>
  )
}
export const useTheme = () => useContext(ThemeContext)
