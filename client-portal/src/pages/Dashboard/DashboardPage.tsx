import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { 
  ChartBarIcon, 
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ClockIcon,
  ServerIcon,
  SignalIcon,
  BoltIcon,
  CurrencyDollarIcon
} from '@heroicons/react/24/outline';

// Components
import StatsCard from '../../components/Dashboard/StatsCard';
import KPIChart from '../../components/Dashboard/KPIChart';
import AlertsList from '../../components/Dashboard/AlertsList';
import SiteMap from '../../components/Dashboard/SiteMap';
import RecentEvents from '../../components/Dashboard/RecentEvents';
import MaintenanceCalendar from '../../components/Dashboard/MaintenanceCalendar';
import LoadingSpinner from '../../components/Common/LoadingSpinner';

// Hooks
import { useTenant } from '../../contexts/TenantContext';
import { useAuth } from '../../contexts/AuthContext';

// API functions
import { fetchDashboardData, fetchKPIMetrics, fetchActiveAlerts, fetchSiteOverview } from '../../services/api';

const DashboardPage: React.FC = () => {
  const { tenant } = useTenant();
  const { user } = useAuth();
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h');

  // Dashboard data queries
  const { data: dashboardData, isLoading: isDashboardLoading } = useQuery({
    queryKey: ['dashboard', selectedTimeRange],
    queryFn: () => fetchDashboardData(selectedTimeRange),
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  const { data: kpiMetrics, isLoading: isKPILoading } = useQuery({
    queryKey: ['kpi-metrics', selectedTimeRange],
    queryFn: () => fetchKPIMetrics({ time_range: selectedTimeRange, limit: 100 }),
    refetchInterval: 60000, // Refresh every minute
  });

  const { data: activeAlerts, isLoading: isAlertsLoading } = useQuery({
    queryKey: ['active-alerts'],
    queryFn: () => fetchActiveAlerts(),
    refetchInterval: 15000, // Refresh every 15 seconds
  });

  const { data: siteOverview, isLoading: isSitesLoading } = useQuery({
    queryKey: ['site-overview'],
    queryFn: () => fetchSiteOverview(),
    refetchInterval: 300000, // Refresh every 5 minutes
  });

  if (isDashboardLoading || isKPILoading || isAlertsLoading || isSitesLoading) {
    return <LoadingSpinner />;
  }

  // Calculate KPI averages by category
  const getKPIAverageByCategory = (category: string) => {
    if (!kpiMetrics?.length) return 0;
    const categoryKPIs = kpiMetrics.filter(kpi => kpi.category === category);
    if (categoryKPIs.length === 0) return 0;
    return categoryKPIs.reduce((sum, kpi) => sum + kpi.current_value, 0) / categoryKPIs.length;
  };

  const networkAvg = getKPIAverageByCategory('network');
  const energyAvg = getKPIAverageByCategory('energy');
  const operationalAvg = getKPIAverageByCategory('operational');
  const financialAvg = getKPIAverageByCategory('financial');

  // Count alerts by severity
  const alertsBySeverity = activeAlerts?.reduce((acc: any, alert: any) => {
    acc[alert.severity] = (acc[alert.severity] || 0) + 1;
    return acc;
  }, {}) || {};

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">
            Welcome back, {user?.name}
          </h1>
          <p className="text-gray-600">
            {tenant?.branding.companyName} Operations Dashboard
          </p>
        </div>
        
        <div className="flex items-center space-x-2">
          <select
            value={selectedTimeRange}
            onChange={(e) => setSelectedTimeRange(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="1h">Last Hour</option>
            <option value="6h">Last 6 Hours</option>
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
          </select>
        </div>
      </div>

      {/* Key Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatsCard
          title="Network Performance"
          value={`${networkAvg.toFixed(1)}%`}
          change={`+${((networkAvg - 85) * 0.1).toFixed(1)}%`}
          changeType={networkAvg > 85 ? 'positive' : 'negative'}
          icon={<SignalIcon className="w-6 h-6" />}
          color="blue"
        />
        
        <StatsCard
          title="Energy Efficiency"
          value={`${energyAvg.toFixed(1)}%`}
          change={`+${((energyAvg - 80) * 0.1).toFixed(1)}%`}
          changeType={energyAvg > 80 ? 'positive' : 'negative'}
          icon={<BoltIcon className="w-6 h-6" />}
          color="green"
        />
        
        <StatsCard
          title="Operational Excellence"
          value={`${operationalAvg.toFixed(1)}%`}
          change={`+${((operationalAvg - 90) * 0.1).toFixed(1)}%`}
          changeType={operationalAvg > 90 ? 'positive' : 'negative'}
          icon={<CheckCircleIcon className="w-6 h-6" />}
          color="indigo"
        />
        
        <StatsCard
          title="Financial Performance"
          value={`${financialAvg.toFixed(1)}%`}
          change={`+${((financialAvg - 75) * 0.1).toFixed(1)}%`}
          changeType={financialAvg > 75 ? 'positive' : 'negative'}
          icon={<CurrencyDollarIcon className="w-6 h-6" />}
          color="purple"
        />
      </div>

      {/* Main Dashboard Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* KPI Trends Chart - Takes 2/3 width */}
        <div className="lg:col-span-2">
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-lg font-semibold text-gray-900">KPI Trends</h2>
              <div className="flex space-x-2">
                <div className="flex items-center space-x-1">
                  <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                  <span className="text-sm text-gray-600">Network</span>
                </div>
                <div className="flex items-center space-x-1">
                  <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                  <span className="text-sm text-gray-600">Energy</span>
                </div>
                <div className="flex items-center space-x-1">
                  <div className="w-3 h-3 bg-indigo-500 rounded-full"></div>
                  <span className="text-sm text-gray-600">Operational</span>
                </div>
                <div className="flex items-center space-x-1">
                  <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
                  <span className="text-sm text-gray-600">Financial</span>
                </div>
              </div>
            </div>
            <KPIChart data={kpiMetrics} timeRange={selectedTimeRange} />
          </div>
        </div>

        {/* Active Alerts - Takes 1/3 width */}
        <div className="lg:col-span-1">
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-lg font-semibold text-gray-900">Active Alerts</h2>
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                {activeAlerts?.length || 0} Active
              </span>
            </div>

            {/* Alert Summary */}
            <div className="grid grid-cols-2 gap-3 mb-4">
              <div className="text-center p-2 bg-red-50 rounded">
                <div className="text-lg font-bold text-red-600">
                  {alertsBySeverity.critical || 0}
                </div>
                <div className="text-xs text-red-600">Critical</div>
              </div>
              <div className="text-center p-2 bg-orange-50 rounded">
                <div className="text-lg font-bold text-orange-600">
                  {alertsBySeverity.major || 0}
                </div>
                <div className="text-xs text-orange-600">Major</div>
              </div>
              <div className="text-center p-2 bg-yellow-50 rounded">
                <div className="text-lg font-bold text-yellow-600">
                  {alertsBySeverity.minor || 0}
                </div>
                <div className="text-xs text-yellow-600">Minor</div>
              </div>
              <div className="text-center p-2 bg-blue-50 rounded">
                <div className="text-lg font-bold text-blue-600">
                  {alertsBySeverity.warning || 0}
                </div>
                <div className="text-xs text-blue-600">Warning</div>
              </div>
            </div>

            <AlertsList alerts={activeAlerts?.slice(0, 5) || []} />
            
            {activeAlerts && activeAlerts.length > 5 && (
              <div className="mt-4 text-center">
                <a
                  href="/alerts"
                  className="text-sm text-blue-600 hover:text-blue-800 font-medium"
                >
                  View all alerts ({activeAlerts.length})
                </a>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Secondary Dashboard Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Site Overview Map */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Site Overview</h2>
          <SiteMap sites={siteOverview?.sites || []} />
          
          <div className="mt-4 grid grid-cols-3 gap-4 text-center">
            <div>
              <div className="text-lg font-bold text-green-600">
                {siteOverview?.stats.healthy || 0}
              </div>
              <div className="text-xs text-gray-600">Healthy Sites</div>
            </div>
            <div>
              <div className="text-lg font-bold text-orange-600">
                {siteOverview?.stats.degraded || 0}
              </div>
              <div className="text-xs text-gray-600">Degraded Sites</div>
            </div>
            <div>
              <div className="text-lg font-bold text-red-600">
                {siteOverview?.stats.critical || 0}
              </div>
              <div className="text-xs text-gray-600">Critical Sites</div>
            </div>
          </div>
        </div>

        {/* Recent Events & Maintenance */}
        <div className="space-y-6">
          {/* Recent Events */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Recent Events</h2>
            <RecentEvents limit={5} />
          </div>

          {/* Upcoming Maintenance */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Upcoming Maintenance</h2>
            <MaintenanceCalendar compact={true} />
          </div>
        </div>
      </div>

      {/* Real-time Status Bar */}
      <div className="bg-white rounded-lg shadow-md p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
              <span className="text-sm font-medium text-gray-900">System Status: Operational</span>
            </div>
            <div className="text-sm text-gray-600">
              Last updated: {new Date().toLocaleTimeString()}
            </div>
          </div>

          <div className="flex items-center space-x-6 text-sm text-gray-600">
            <div className="flex items-center space-x-1">
              <ServerIcon className="w-4 h-4" />
              <span>{siteOverview?.stats.total || 0} Sites</span>
            </div>
            <div className="flex items-center space-x-1">
              <ExclamationTriangleIcon className="w-4 h-4" />
              <span>{activeAlerts?.length || 0} Alerts</span>
            </div>
            <div className="flex items-center space-x-1">
              <ChartBarIcon className="w-4 h-4" />
              <span>{kpiMetrics?.length || 0} KPIs</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;