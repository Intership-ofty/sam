import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ClockIcon,
  UserGroupIcon,
  ServerIcon,
  ChartBarIcon,
  BoltIcon,
  ArrowTrendingUpIcon,
  MapPinIcon,
  FireIcon
} from '@heroicons/react/24/outline';

// Components
import StatsCard from '../../components/Dashboard/StatsCard';
import IncidentMap from '../../components/NOC/IncidentMap';
import IncidentTable from '../../components/NOC/IncidentTable';
import TeamPerformance from '../../components/NOC/TeamPerformance';
import EscalationMetrics from '../../components/NOC/EscalationMetrics';
import SiteHealthGrid from '../../components/NOC/SiteHealthGrid';
import RealTimeAlerts from '../../components/NOC/RealTimeAlerts';
import LoadingSpinner from '../../components/Common/LoadingSpinner';

// Hooks
import { useTenant } from '../../contexts/TenantContext';
import { useAuth } from '../../contexts/AuthContext';

// API functions
import { fetchNOCDashboard, fetchActiveIncidents } from '../../services/api';

const NOCDashboard: React.FC = () => {
  const { tenant } = useTenant();
  const { user, checkPermission } = useAuth();
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h');
  const [autoRefresh, setAutoRefresh] = useState(true);

  // NOC dashboard data
  const { data: nocData, isLoading: isNOCLoading, refetch: refetchNOC } = useQuery({
    queryKey: ['noc-dashboard', selectedTimeRange],
    queryFn: () => fetchNOCDashboard({ time_range: selectedTimeRange }),
    refetchInterval: autoRefresh ? 30000 : false, // Refresh every 30 seconds
  });

  const { data: activeIncidents, isLoading: isIncidentsLoading } = useQuery({
    queryKey: ['active-incidents'],
    queryFn: () => fetchActiveIncidents(),
    refetchInterval: autoRefresh ? 15000 : false, // Refresh every 15 seconds
  });

  // Auto-refresh control
  useEffect(() => {
    const handleVisibilityChange = () => {
      setAutoRefresh(!document.hidden);
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => document.removeEventListener('visibilitychange', handleVisibilityChange);
  }, []);

  if (isNOCLoading || isIncidentsLoading) {
    return <LoadingSpinner />;
  }

  const incidentStats = nocData?.incident_statistics || {};
  const siteHealth = nocData?.site_health || [];
  const recentIncidents = nocData?.recent_incidents || [];
  const teamPerformance = nocData?.team_performance || [];
  const escalationMetrics = nocData?.escalation_metrics || {};

  // Calculate health percentages
  const totalSites = siteHealth.length;
  const healthySites = siteHealth.filter(site => site.status === 'healthy').length;
  const criticalSites = siteHealth.filter(site => site.status === 'critical').length;
  const warningSites = siteHealth.filter(site => site.status === 'warning').length;
  const healthPercentage = totalSites > 0 ? (healthySites / totalSites) * 100 : 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">
            Network Operations Center
          </h1>
          <p className="text-gray-600">
            Real-time operational monitoring and incident management
          </p>
        </div>
        
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              id="auto-refresh"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <label htmlFor="auto-refresh" className="text-sm text-gray-600">
              Auto-refresh
            </label>
          </div>

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

          <button
            onClick={() => refetchNOC()}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Refresh
          </button>
        </div>
      </div>

      {/* Key Operational Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-4">
        <StatsCard
          title="Active Incidents"
          value={incidentStats.active?.toString() || "0"}
          change={`${incidentStats.in_time_range || 0} new`}
          changeType={incidentStats.active > 10 ? 'negative' : 'neutral'}
          icon={<ExclamationTriangleIcon className="w-5 h-5" />}
          color="orange"
          compact
        />
        
        <StatsCard
          title="Critical Incidents"
          value={incidentStats.critical?.toString() || "0"}
          change={`${escalationMetrics.recent || 0} escalated`}
          changeType={incidentStats.critical > 0 ? 'negative' : 'positive'}
          icon={<FireIcon className="w-5 h-5" />}
          color="red"
          compact
        />
        
        <StatsCard
          title="Site Health"
          value={`${healthPercentage.toFixed(1)}%`}
          change={`${healthySites}/${totalSites} healthy`}
          changeType={healthPercentage >= 95 ? 'positive' : healthPercentage >= 85 ? 'neutral' : 'negative'}
          icon={<ServerIcon className="w-5 h-5" />}
          color="green"
          compact
        />
        
        <StatsCard
          title="MTTR"
          value={`${incidentStats.avg_resolution_hours?.toFixed(1) || '0.0'}h`}
          change="Average"
          changeType={incidentStats.avg_resolution_hours < 4 ? 'positive' : 'neutral'}
          icon={<ClockIcon className="w-5 h-5" />}
          color="blue"
          compact
        />
        
        <StatsCard
          title="Team Utilization"
          value={`${teamPerformance.length || 0}`}
          change="Engineers active"
          changeType="neutral"
          icon={<UserGroupIcon className="w-5 h-5" />}
          color="indigo"
          compact
        />
        
        <StatsCard
          title="Escalations"
          value={escalationMetrics.total?.toString() || "0"}
          change={`${escalationMetrics.avg_time_minutes?.toFixed(0) || 0}m avg`}
          changeType={escalationMetrics.recent > 5 ? 'negative' : 'neutral'}
          icon={<ArrowTrendingUpIcon className="w-5 h-5" />}
          color="purple"
          compact
        />
      </div>

      {/* Real-time Status Bar */}
      <div className="bg-white rounded-lg shadow-md p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-6">
            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full animate-pulse ${
                incidentStats.critical > 0 ? 'bg-red-500' : 
                incidentStats.active > 5 ? 'bg-yellow-500' : 'bg-green-500'
              }`}></div>
              <span className="text-sm font-medium text-gray-900">
                System Status: {
                  incidentStats.critical > 0 ? 'Critical' : 
                  incidentStats.active > 5 ? 'Degraded' : 'Operational'
                }
              </span>
            </div>
            <div className="text-sm text-gray-600">
              Last updated: {nocData?.last_updated ? new Date(nocData.last_updated).toLocaleTimeString() : 'Never'}
            </div>
          </div>

          <div className="flex items-center space-x-6 text-sm text-gray-600">
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-red-500 rounded-full"></div>
              <span>{criticalSites} Critical</span>
            </div>
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
              <span>{warningSites} Warning</span>
            </div>
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              <span>{healthySites} Healthy</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main NOC Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Site Health Overview - Takes 2/3 width */}
        <div className="lg:col-span-2">
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-lg font-semibold text-gray-900">Site Health Overview</h2>
              <div className="flex space-x-2">
                <button className="px-3 py-1 text-xs bg-gray-100 text-gray-600 rounded hover:bg-gray-200">
                  Map View
                </button>
                <button className="px-3 py-1 text-xs bg-blue-100 text-blue-600 rounded">
                  Grid View
                </button>
              </div>
            </div>
            <SiteHealthGrid sites={siteHealth} />
          </div>
        </div>

        {/* Real-time Alerts - Takes 1/3 width */}
        <div className="lg:col-span-1">
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-lg font-semibold text-gray-900">Real-time Alerts</h2>
              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                {activeIncidents?.length || 0} Active
              </span>
            </div>
            <RealTimeAlerts incidents={activeIncidents?.slice(0, 8) || []} />
          </div>
        </div>
      </div>

      {/* Secondary Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Incident Management */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-lg font-semibold text-gray-900">Recent Incidents</h2>
            <a
              href="/noc/incidents"
              className="text-sm text-blue-600 hover:text-blue-800 font-medium"
            >
              View All →
            </a>
          </div>
          <IncidentTable 
            incidents={recentIncidents.slice(0, 6)} 
            compact={true}
            showActions={checkPermission('incidents.update')}
          />
        </div>

        {/* Team Performance */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-lg font-semibold text-gray-900">Team Performance</h2>
            <div className="text-sm text-gray-500">
              {selectedTimeRange === '24h' ? 'Last 24 Hours' : 
               selectedTimeRange === '7d' ? 'Last 7 Days' : 
               selectedTimeRange === '30d' ? 'Last 30 Days' : 'Current Period'}
            </div>
          </div>
          <TeamPerformance engineers={teamPerformance} />
        </div>
      </div>

      {/* Escalation and SLA Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Escalation Metrics */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Escalation Metrics</h2>
          <EscalationMetrics 
            metrics={escalationMetrics}
            incidents={recentIncidents}
            timeRange={selectedTimeRange}
          />
        </div>

        {/* Geographic Incident Distribution */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Incident Distribution</h2>
          <IncidentMap 
            incidents={activeIncidents || []}
            sites={siteHealth}
          />
        </div>
      </div>

      {/* Critical Alerts Banner */}
      {incidentStats.critical > 0 && (
        <div className="fixed bottom-4 right-4 max-w-sm">
          <div className="bg-red-600 text-white rounded-lg shadow-lg p-4">
            <div className="flex items-center space-x-2">
              <FireIcon className="w-5 h-5" />
              <div>
                <div className="font-semibold">Critical Incidents Active</div>
                <div className="text-sm opacity-90">
                  {incidentStats.critical} incident{incidentStats.critical !== 1 ? 's' : ''} require immediate attention
                </div>
              </div>
            </div>
            <div className="mt-2">
              <a
                href="/noc/incidents?severity=critical"
                className="text-sm underline hover:no-underline"
              >
                View Details →
              </a>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default NOCDashboard;