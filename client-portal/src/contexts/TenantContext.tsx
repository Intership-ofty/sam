import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { useAuth } from './AuthContext';

export interface Tenant {
  id: string;
  name: string;
  displayName: string;
  domain: string;
  logo?: string;
  primaryColor: string;
  settings: {
    timezone: string;
    dateFormat: string;
    currency: string;
    language: string;
  };
  subscription: {
    plan: string;
    features: string[];
    limits: {
      sites: number;
      users: number;
      storage: number;
    };
  };
  branding: {
    companyName: string;
    logoUrl?: string;
    primaryColor: string;
    secondaryColor: string;
  };
}

interface TenantContextType {
  tenant: Tenant | null;
  isLoading: boolean;
  refreshTenant: () => Promise<void>;
  hasFeature: (feature: string) => boolean;
  isWithinLimit: (resource: string, currentUsage: number) => boolean;
}

const TenantContext = createContext<TenantContextType | undefined>(undefined);

interface TenantProviderProps {
  children: ReactNode;
}

export const useTenant = (): TenantContextType => {
  const context = useContext(TenantContext);
  if (!context) {
    throw new Error('useTenant must be used within a TenantProvider');
  }
  return context;
};

export const TenantProvider: React.FC<TenantProviderProps> = ({ children }) => {
  const [tenant, setTenant] = useState<Tenant | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const { user, isAuthenticated } = useAuth();

  // Load tenant data when user is authenticated
  useEffect(() => {
    if (isAuthenticated && user?.tenantId) {
      loadTenantData(user.tenantId);
    }
  }, [isAuthenticated, user?.tenantId]);

  const loadTenantData = async (tenantId: string) => {
    try {
      setIsLoading(true);

      const token = localStorage.getItem('accessToken');
      const response = await fetch(`/api/v1/tenants/${tenantId}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        const tenantData = await response.json();
        setTenant(tenantData);

        // Apply tenant branding
        applyTenantBranding(tenantData.branding);
      } else {
        console.error('Failed to load tenant data');
      }
    } catch (error) {
      console.error('Error loading tenant data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const applyTenantBranding = (branding: Tenant['branding']) => {
    // Apply primary color to CSS custom properties
    const root = document.documentElement;
    root.style.setProperty('--tenant-primary-color', branding.primaryColor);
    root.style.setProperty('--tenant-secondary-color', branding.secondaryColor);

    // Update page title
    document.title = `${branding.companyName} - Towerco AIOps`;

    // Update favicon if logo is available
    if (branding.logoUrl) {
      const favicon = document.querySelector('link[rel="icon"]') as HTMLLinkElement;
      if (favicon) {
        favicon.href = branding.logoUrl;
      }
    }
  };

  const refreshTenant = async (): Promise<void> => {
    if (user?.tenantId) {
      await loadTenantData(user.tenantId);
    }
  };

  const hasFeature = (feature: string): boolean => {
    if (!tenant) return false;
    return tenant.subscription.features.includes(feature);
  };

  const isWithinLimit = (resource: string, currentUsage: number): boolean => {
    if (!tenant) return false;
    
    const limits = tenant.subscription.limits;
    const limit = limits[resource as keyof typeof limits];
    
    if (limit === -1) return true; // Unlimited
    return currentUsage < limit;
  };

  const contextValue: TenantContextType = {
    tenant,
    isLoading,
    refreshTenant,
    hasFeature,
    isWithinLimit,
  };

  return (
    <TenantContext.Provider value={contextValue}>
      {children}
    </TenantContext.Provider>
  );
};