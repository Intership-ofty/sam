import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { Toaster } from 'react-hot-toast';

// Components
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { TenantProvider } from './contexts/TenantContext';
import { ThemeProvider } from './contexts/ThemeContext';
import Sidebar from './components/Layout/Sidebar';
import Header from './components/Layout/Header';
import LoadingSpinner from './components/Common/LoadingSpinner';

// Pages
import LoginPage from './pages/Auth/LoginPage';
import DashboardPage from './pages/Dashboard/DashboardPage';
import SitesPage from './pages/Sites/SitesPage';
import SiteDetailPage from './pages/Sites/SiteDetailPage';
import KPIsPage from './pages/KPIs/KPIsPage';
import AlertsPage from './pages/Alerts/AlertsPage';
import ReportsPage from './pages/Reports/ReportsPage';
import SLAPage from './pages/SLA/SLAPage';
import MaintenancePage from './pages/Maintenance/MaintenancePage';
import SettingsPage from './pages/Settings/SettingsPage';

// Styles
import './App.css';

// Create Query Client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      staleTime: 5 * 60 * 1000, // 5 minutes
      refetchOnWindowFocus: false,
    },
  },
});

// Protected Route Component
const ProtectedRoute: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return <LoadingSpinner />;
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
};

// Main Layout Component
const Layout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <div className="flex h-screen bg-gray-50">
      <Sidebar isOpen={sidebarOpen} onToggle={() => setSidebarOpen(!sidebarOpen)} />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header onMenuClick={() => setSidebarOpen(!sidebarOpen)} />
        <main className="flex-1 overflow-x-hidden overflow-y-auto bg-gray-50 p-6">
          {children}
        </main>
      </div>
    </div>
  );
};

// Main App Component
const App: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <AuthProvider>
          <TenantProvider>
            <Router>
              <div className="App">
                <Routes>
                  {/* Public Routes */}
                  <Route path="/login" element={<LoginPage />} />

                  {/* Protected Routes */}
                  <Route
                    path="/"
                    element={
                      <ProtectedRoute>
                        <Layout>
                          <Navigate to="/dashboard" replace />
                        </Layout>
                      </ProtectedRoute>
                    }
                  />

                  <Route
                    path="/dashboard"
                    element={
                      <ProtectedRoute>
                        <Layout>
                          <DashboardPage />
                        </Layout>
                      </ProtectedRoute>
                    }
                  />

                  <Route
                    path="/sites"
                    element={
                      <ProtectedRoute>
                        <Layout>
                          <SitesPage />
                        </Layout>
                      </ProtectedRoute>
                    }
                  />

                  <Route
                    path="/sites/:siteId"
                    element={
                      <ProtectedRoute>
                        <Layout>
                          <SiteDetailPage />
                        </Layout>
                      </ProtectedRoute>
                    }
                  />

                  <Route
                    path="/kpis"
                    element={
                      <ProtectedRoute>
                        <Layout>
                          <KPIsPage />
                        </Layout>
                      </ProtectedRoute>
                    }
                  />

                  <Route
                    path="/alerts"
                    element={
                      <ProtectedRoute>
                        <Layout>
                          <AlertsPage />
                        </Layout>
                      </ProtectedRoute>
                    }
                  />

                  <Route
                    path="/reports"
                    element={
                      <ProtectedRoute>
                        <Layout>
                          <ReportsPage />
                        </Layout>
                      </ProtectedRoute>
                    }
                  />

                  <Route
                    path="/sla"
                    element={
                      <ProtectedRoute>
                        <Layout>
                          <SLAPage />
                        </Layout>
                      </ProtectedRoute>
                    }
                  />

                  <Route
                    path="/maintenance"
                    element={
                      <ProtectedRoute>
                        <Layout>
                          <MaintenancePage />
                        </Layout>
                      </ProtectedRoute>
                    }
                  />

                  <Route
                    path="/settings"
                    element={
                      <ProtectedRoute>
                        <Layout>
                          <SettingsPage />
                        </Layout>
                      </ProtectedRoute>
                    }
                  />

                  {/* Catch all route */}
                  <Route
                    path="*"
                    element={
                      <ProtectedRoute>
                        <Layout>
                          <Navigate to="/dashboard" replace />
                        </Layout>
                      </ProtectedRoute>
                    }
                  />
                </Routes>

                {/* Global Components */}
                <Toaster
                  position="top-right"
                  toastOptions={{
                    duration: 4000,
                    style: {
                      background: '#363636',
                      color: '#fff',
                    },
                    success: {
                      duration: 3000,
                      theme: {
                        primary: '#4ade80',
                        secondary: '#000',
                      },
                    },
                  }}
                />
              </div>
            </Router>
          </TenantProvider>
        </AuthProvider>
      </ThemeProvider>

      {/* React Query Devtools */}
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  );
};

export default App;