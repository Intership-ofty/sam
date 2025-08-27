# ADR-005: React TypeScript for Frontend

## Status
Accepted

## Context
The Towerco AIOps platform requires a sophisticated frontend application with the following requirements:

- **Real-time Data Visualization**: Dynamic dashboards with live KPI updates
- **Complex State Management**: Multi-tenant context, user preferences, and application state
- **Responsive Design**: Support for desktop, tablet, and mobile viewports
- **Type Safety**: Strong typing to prevent runtime errors in critical operations
- **Developer Experience**: Fast development, debugging, and testing capabilities
- **Performance**: Efficient rendering of large datasets and real-time updates
- **Accessibility**: WCAG compliance for enterprise users
- **Internationalization**: Support for multiple languages and locales

The application needs to handle complex interactions like real-time dashboards, incident management workflows, and business intelligence visualizations.

## Decision
We will build the frontend application using **React 18** with **TypeScript**, leveraging modern React patterns and a carefully selected ecosystem of libraries.

### Technology Stack:
- **React 18** - Core UI framework with concurrent features
- **TypeScript** - Static typing and enhanced developer experience
- **Tailwind CSS** - Utility-first CSS framework for consistent styling
- **React Query/TanStack Query** - Server state management and caching
- **React Router** - Client-side routing and navigation
- **Zustand** - Lightweight client-side state management
- **React Hook Form** - Performant form handling with validation
- **Chart.js/Recharts** - Data visualization and charting
- **React Hot Toast** - User notifications and feedback

## Alternatives Considered

### 1. Vue.js 3 with TypeScript
- **Pros**: Excellent TypeScript support, simpler learning curve, great performance
- **Cons**: Smaller ecosystem, less enterprise adoption, fewer specialized libraries

### 2. Angular 15+
- **Pros**: Built-in TypeScript, comprehensive framework, enterprise-focused
- **Cons**: Steep learning curve, heavy bundle size, complex architecture

### 3. Svelte/SvelteKit
- **Pros**: Excellent performance, small bundle size, modern features
- **Cons**: Smaller ecosystem, less mature, limited enterprise tooling

### 4. Next.js (React Framework)
- **Pros**: Full-stack capabilities, excellent performance, great DX
- **Cons**: Overkill for SPA, server-side complexity, vendor lock-in

## Consequences

### Positive
- **Strong Ecosystem**: Vast library ecosystem and community support
- **Type Safety**: TypeScript prevents many runtime errors
- **Developer Experience**: Excellent tooling, debugging, and hot reload
- **Performance**: React 18 concurrent features and optimizations
- **Talent Pool**: Large pool of React developers available
- **Component Reusability**: Modular component architecture
- **Testing**: Mature testing ecosystem with Jest and Testing Library

### Negative
- **Bundle Size**: React ecosystem can lead to larger bundles
- **Configuration Complexity**: More initial setup compared to simpler frameworks
- **Rapid Ecosystem Changes**: Need to stay current with frequent updates
- **Learning Curve**: TypeScript adds complexity for junior developers

## Implementation

### Project Structure
```
client-portal/
├── public/
│   ├── index.html
│   └── favicon.ico
├── src/
│   ├── components/          # Reusable UI components
│   │   ├── Common/         # Generic components
│   │   ├── Dashboard/      # Dashboard-specific components
│   │   ├── NOC/           # NOC-specific components
│   │   └── SLA/           # SLA-specific components
│   ├── contexts/          # React Context providers
│   │   ├── AuthContext.tsx
│   │   ├── TenantContext.tsx
│   │   └── ThemeContext.tsx
│   ├── hooks/             # Custom React hooks
│   │   ├── useAuth.ts
│   │   ├── useWebSocket.ts
│   │   └── useLocalStorage.ts
│   ├── pages/             # Page components
│   │   ├── Dashboard/
│   │   ├── NOC/
│   │   └── SLA/
│   ├── services/          # API and external services
│   │   ├── api.ts
│   │   ├── websocket.ts
│   │   └── storage.ts
│   ├── types/             # TypeScript type definitions
│   │   ├── api.ts
│   │   ├── auth.ts
│   │   └── tenant.ts
│   ├── utils/             # Utility functions
│   │   ├── formatting.ts
│   │   ├── validation.ts
│   │   └── constants.ts
│   ├── App.tsx            # Main application component
│   └── index.tsx          # Application entry point
├── package.json
├── tsconfig.json
└── tailwind.config.js
```

### TypeScript Configuration
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "lib": ["DOM", "DOM.Iterable", "ES2020"],
    "allowJs": true,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "strict": true,
    "forceConsistentCasingInFileNames": true,
    "moduleResolution": "node",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "baseUrl": "src",
    "paths": {
      "@/*": ["*"],
      "@/components/*": ["components/*"],
      "@/hooks/*": ["hooks/*"],
      "@/services/*": ["services/*"],
      "@/types/*": ["types/*"],
      "@/utils/*": ["utils/*"]
    }
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules"]
}
```

### Type Definitions Strategy
```typescript
// types/api.ts - API response types
export interface APIResponse<T> {
  data: T;
  success: boolean;
  message?: string;
  errors?: string[];
}

export interface PaginatedResponse<T> {
  items: T[];
  total_count: number;
  page: number;
  page_size: number;
  has_more: boolean;
}

// types/tenant.ts - Business domain types
export interface Tenant {
  id: string;
  name: string;
  displayName: string;
  domain: string;
  settings: TenantSettings;
  subscription: TenantSubscription;
  branding: TenantBranding;
}

export interface TenantSettings {
  timezone: string;
  dateFormat: string;
  currency: string;
  language: string;
}

// types/auth.ts - Authentication types
export interface User {
  id: string;
  email: string;
  name: string;
  role: UserRole;
  tenantId: string;
  tenantName: string;
  permissions: string[];
  lastLogin?: string;
}

export enum UserRole {
  ADMIN = 'admin',
  MANAGER = 'manager',
  OPERATOR = 'operator',
  VIEWER = 'viewer'
}
```

### State Management Architecture

#### Global State with Context
```typescript
// contexts/AuthContext.tsx
interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<boolean>;
  logout: () => void;
  refreshToken: () => Promise<boolean>;
  checkPermission: (permission: string) => boolean;
}

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Implementation...

  const contextValue: AuthContextType = {
    user,
    isAuthenticated: !!user,
    isLoading,
    login,
    logout,
    refreshToken,
    checkPermission,
  };

  return (
    <AuthContext.Provider value={contextValue}>
      {children}
    </AuthContext.Provider>
  );
};
```

#### Server State with React Query
```typescript
// hooks/useDashboardData.ts
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { fetchDashboardData } from '@/services/api';

export const useDashboardData = (timeRange: string) => {
  const queryClient = useQueryClient();

  return useQuery({
    queryKey: ['dashboard', timeRange],
    queryFn: () => fetchDashboardData(timeRange),
    refetchInterval: 30000, // Refresh every 30 seconds
    staleTime: 25000, // Consider data stale after 25 seconds
    cacheTime: 300000, // Keep in cache for 5 minutes
    onError: (error) => {
      console.error('Dashboard data fetch failed:', error);
      toast.error('Failed to load dashboard data');
    },
  });
};

// Prefetching for performance
export const usePrefetchDashboard = () => {
  const queryClient = useQueryClient();

  return (timeRange: string) => {
    queryClient.prefetchQuery({
      queryKey: ['dashboard', timeRange],
      queryFn: () => fetchDashboardData(timeRange),
      staleTime: 10000,
    });
  };
};
```

### Component Architecture

#### Compound Component Pattern
```typescript
// components/Dashboard/StatsCard.tsx
interface StatsCardProps {
  title: string;
  value: string | number;
  change?: string;
  changeType?: 'positive' | 'negative' | 'neutral';
  icon?: React.ReactNode;
  color?: 'blue' | 'green' | 'red' | 'yellow';
  onClick?: () => void;
}

export const StatsCard: React.FC<StatsCardProps> = ({
  title,
  value,
  change,
  changeType = 'neutral',
  icon,
  color = 'blue',
  onClick
}) => {
  const colorClasses = {
    blue: 'bg-blue-50 text-blue-700 border-blue-200',
    green: 'bg-green-50 text-green-700 border-green-200',
    red: 'bg-red-50 text-red-700 border-red-200',
    yellow: 'bg-yellow-50 text-yellow-700 border-yellow-200',
  };

  const changeClasses = {
    positive: 'text-green-600',
    negative: 'text-red-600',
    neutral: 'text-gray-600',
  };

  return (
    <div
      className={`p-6 rounded-lg border ${colorClasses[color]} ${
        onClick ? 'cursor-pointer hover:shadow-md transition-shadow' : ''
      }`}
      onClick={onClick}
    >
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-2xl font-bold text-gray-900">{value}</p>
          {change && (
            <p className={`text-sm ${changeClasses[changeType]}`}>
              {change}
            </p>
          )}
        </div>
        {icon && (
          <div className="flex-shrink-0 ml-4">
            {icon}
          </div>
        )}
      </div>
    </div>
  );
};
```

#### Custom Hooks for Business Logic
```typescript
// hooks/useRealTimeKPIs.ts
import { useState, useEffect, useCallback } from 'react';
import { useWebSocket } from './useWebSocket';

interface KPIUpdate {
  siteId: string;
  metricName: string;
  value: number;
  timestamp: string;
}

export const useRealTimeKPIs = (tenantId: string, siteIds: string[]) => {
  const [kpis, setKPIs] = useState<Map<string, KPIUpdate>>(new Map());
  const [isConnected, setIsConnected] = useState(false);

  const handleKPIUpdate = useCallback((data: KPIUpdate) => {
    setKPIs(prev => {
      const newKPIs = new Map(prev);
      const key = `${data.siteId}:${data.metricName}`;
      newKPIs.set(key, data);
      return newKPIs;
    });
  }, []);

  const { sendMessage, lastMessage, readyState } = useWebSocket(
    `ws://localhost:8000/api/v1/kpis/ws/${tenantId}`,
    {
      onMessage: handleKPIUpdate,
      shouldReconnect: () => true,
      reconnectAttempts: 10,
      reconnectInterval: 3000,
    }
  );

  useEffect(() => {
    setIsConnected(readyState === WebSocket.OPEN);
  }, [readyState]);

  useEffect(() => {
    if (isConnected && siteIds.length > 0) {
      sendMessage(JSON.stringify({
        action: 'subscribe',
        siteIds: siteIds
      }));
    }
  }, [isConnected, siteIds, sendMessage]);

  const getKPI = useCallback((siteId: string, metricName: string): KPIUpdate | undefined => {
    return kpis.get(`${siteId}:${metricName}`);
  }, [kpis]);

  const getLatestKPIs = useCallback((): KPIUpdate[] => {
    return Array.from(kpis.values()).sort(
      (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    );
  }, [kpis]);

  return {
    kpis: Array.from(kpis.values()),
    isConnected,
    getKPI,
    getLatestKPIs
  };
};
```

### Performance Optimization

#### Code Splitting and Lazy Loading
```typescript
// App.tsx - Route-based code splitting
import { lazy, Suspense } from 'react';
import LoadingSpinner from '@/components/Common/LoadingSpinner';

const DashboardPage = lazy(() => import('@/pages/Dashboard/DashboardPage'));
const NOCPage = lazy(() => import('@/pages/NOC/NOCPage'));
const SLAPage = lazy(() => import('@/pages/SLA/SLAPage'));

function App() {
  return (
    <Routes>
      <Route
        path="/dashboard"
        element={
          <Suspense fallback={<LoadingSpinner />}>
            <DashboardPage />
          </Suspense>
        }
      />
      <Route
        path="/noc"
        element={
          <Suspense fallback={<LoadingSpinner />}>
            <NOCPage />
          </Suspense>
        }
      />
      {/* More routes... */}
    </Routes>
  );
}
```

#### Memoization and React.memo
```typescript
// components/Dashboard/KPIChart.tsx
import React, { memo, useMemo } from 'react';

interface KPIChartProps {
  data: KPIData[];
  timeRange: string;
  height?: number;
}

export const KPIChart = memo<KPIChartProps>(({ data, timeRange, height = 300 }) => {
  // Expensive computation memoized
  const chartData = useMemo(() => {
    return processChartData(data, timeRange);
  }, [data, timeRange]);

  const chartOptions = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
      },
    },
    scales: {
      y: {
        beginAtZero: true,
      },
    },
  }), []);

  if (!data?.length) {
    return <div>No data available</div>;
  }

  return (
    <div style={{ height: `${height}px` }}>
      <Line data={chartData} options={chartOptions} />
    </div>
  );
});

KPIChart.displayName = 'KPIChart';
```

### Error Boundary Implementation
```typescript
// components/Common/ErrorBoundary.tsx
import React, { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Uncaught error:', error, errorInfo);
    
    // Report to error tracking service
    if (typeof window !== 'undefined' && window.gtag) {
      window.gtag('event', 'exception', {
        description: error.message,
        fatal: false,
      });
    }
  }

  public render() {
    if (this.state.hasError) {
      return (
        this.props.fallback || (
          <div className="min-h-screen flex items-center justify-center">
            <div className="text-center">
              <h1 className="text-2xl font-bold text-gray-900 mb-4">
                Something went wrong
              </h1>
              <p className="text-gray-600 mb-6">
                We're sorry, but something unexpected happened.
              </p>
              <button
                className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
                onClick={() => window.location.reload()}
              >
                Reload Page
              </button>
            </div>
          </div>
        )
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
```

### Testing Strategy

#### Component Testing with Testing Library
```typescript
// __tests__/components/StatsCard.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { StatsCard } from '@/components/Dashboard/StatsCard';

describe('StatsCard', () => {
  it('renders basic stats card correctly', () => {
    render(
      <StatsCard
        title="Test Metric"
        value="100"
        change="+5%"
        changeType="positive"
      />
    );

    expect(screen.getByText('Test Metric')).toBeInTheDocument();
    expect(screen.getByText('100')).toBeInTheDocument();
    expect(screen.getByText('+5%')).toBeInTheDocument();
  });

  it('calls onClick handler when clickable', () => {
    const handleClick = jest.fn();
    
    render(
      <StatsCard
        title="Clickable Card"
        value="200"
        onClick={handleClick}
      />
    );

    fireEvent.click(screen.getByText('Clickable Card'));
    expect(handleClick).toHaveBeenCalledTimes(1);
  });
});
```

## Build and Deployment

### Build Configuration
```json
{
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject",
    "type-check": "tsc --noEmit",
    "lint": "eslint src --ext .ts,.tsx",
    "lint:fix": "eslint src --ext .ts,.tsx --fix"
  }
}
```

### Performance Bundle Analysis
```bash
# Analyze bundle size
npx webpack-bundle-analyzer build/static/js/*.js

# Performance audit
npm run build && serve -s build
# Run Lighthouse audit
```

## Review Date
January 2025

## References
- [React Documentation](https://react.dev/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [React Query Documentation](https://tanstack.com/query/latest)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)