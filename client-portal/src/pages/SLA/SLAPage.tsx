import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  ChartBarIcon,
  ClockIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  ArrowTrendingUpIcon,
  CalendarIcon,
  DocumentTextIcon
} from '@heroicons/react/24/outline';

// Components
import StatsCard from '../../components/Dashboard/StatsCard';
import SLAChart from '../../components/SLA/SLAChart';
import SLATable from '../../components/SLA/SLATable';
import IncidentTimeline from '../../components/SLA/IncidentTimeline';
import SLATargetEditor from '../../components/SLA/SLATargetEditor';
import LoadingSpinner from '../../components/Common/LoadingSpinner';
import ExportButton from '../../components/Common/ExportButton';

// Hooks
import { useTenant } from '../../contexts/TenantContext';
import { useAuth } from '../../contexts/AuthContext';

// API functions
import { fetchSLAMetrics, fetchSLAIncidents, fetchSLATargets, exportSLAReport } from '../../services/api';

interface SLAMetric {
  metric_name: string;
  target_percentage: number;
  current_percentage: number;
  incidents_count: number;
  downtime_minutes: number;
  availability_percentage: number;
  mttr_minutes: number;
  mtbf_hours: number;
  compliance_status: 'compliant' | 'at_risk' | 'breach';
  period_start: string;
  period_end: string;
}

const SLAPage: React.FC = () => {
  const { tenant } = useTenant();
  const { user, checkPermission } = useAuth();
  const [selectedPeriod, setSelectedPeriod] = useState('current_month');
  const [selectedSites, setSelectedSites] = useState<string[]>([]);
  const [showTargetEditor, setShowTargetEditor] = useState(false);

  // SLA data queries
  const { data: slaMetrics, isLoading: isSLALoading } = useQuery({
    queryKey: ['sla-metrics', selectedPeriod, selectedSites],
    queryFn: () => fetchSLAMetrics({ 
      period: selectedPeriod, 
      sites: selectedSites.length > 0 ? selectedSites : undefined 
    }),
    refetchInterval: 300000, // Refresh every 5 minutes
  });

  const { data: slaIncidents, isLoading: isIncidentsLoading } = useQuery({
    queryKey: ['sla-incidents', selectedPeriod],
    queryFn: () => fetchSLAIncidents({ period: selectedPeriod }),
    refetchInterval: 300000,
  });

  const { data: slaTargets, isLoading: isTargetsLoading } = useQuery({
    queryKey: ['sla-targets'],
    queryFn: () => fetchSLATargets(),
    refetchInterval: 600000, // Refresh every 10 minutes
  });

  if (isSLALoading || isIncidentsLoading || isTargetsLoading) {
    return <LoadingSpinner />;
  }

  // Calculate overall SLA metrics
  const calculateOverallMetrics = () => {
    if (!slaMetrics?.length) return {
      overallAvailability: 0,
      averageMTTR: 0,
      averageMTBF: 0,
      totalIncidents: 0,
      complianceRate: 0
    };

    const totalSites = slaMetrics.length;
    const overallAvailability = slaMetrics.reduce((sum, metric) => sum + metric.availability_percentage, 0) / totalSites;
    const averageMTTR = slaMetrics.reduce((sum, metric) => sum + metric.mttr_minutes, 0) / totalSites;
    const averageMTBF = slaMetrics.reduce((sum, metric) => sum + metric.mtbf_hours, 0) / totalSites;
    const totalIncidents = slaMetrics.reduce((sum, metric) => sum + metric.incidents_count, 0);
    const compliantSites = slaMetrics.filter(metric => metric.compliance_status === 'compliant').length;
    const complianceRate = (compliantSites / totalSites) * 100;

    return {
      overallAvailability,
      averageMTTR,
      averageMTBF,
      totalIncidents,
      complianceRate
    };
  };

  const overallMetrics = calculateOverallMetrics();

  // Count metrics by compliance status
  const complianceBreakdown = slaMetrics?.reduce((acc: any, metric: SLAMetric) => {
    acc[metric.compliance_status] = (acc[metric.compliance_status] || 0) + 1;
    return acc;
  }, {}) || {};

  const handleExportReport = async (format: 'pdf' | 'excel') => {
    try {
      const blob = await exportSLAReport({
        period: selectedPeriod,
        sites: selectedSites,
        format,
        include_charts: true,
        include_incidents: true
      });
      
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `SLA_Report_${selectedPeriod}_${new Date().toISOString().split('T')[0]}.${format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Export failed:', error);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">
            SLA Management & Reporting
          </h1>
          <p className="text-gray-600">
            Service Level Agreement compliance and performance metrics
          </p>
        </div>
        
        <div className="flex items-center space-x-3">
          <select
            value={selectedPeriod}
            onChange={(e) => setSelectedPeriod(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="current_month">Current Month</option>
            <option value="last_month">Last Month</option>
            <option value="current_quarter">Current Quarter</option>
            <option value="last_quarter">Last Quarter</option>
            <option value="current_year">Current Year</option>
            <option value="last_year">Last Year</option>
            <option value="custom">Custom Range</option>
          </select>

          {checkPermission('sla.manage') && (
            <button
              onClick={() => setShowTargetEditor(true)}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              Manage Targets
            </button>
          )}

          <ExportButton
            onExport={handleExportReport}
            formats={['pdf', 'excel']}
            filename="SLA_Report"
          />
        </div>
      </div>

      {/* Key SLA Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
        <StatsCard
          title="Overall Availability"
          value={`${overallMetrics.overallAvailability.toFixed(3)}%`}
          change={`Target: ${slaTargets?.availability_target || 99.9}%`}
          changeType={overallMetrics.overallAvailability >= (slaTargets?.availability_target || 99.9) ? 'positive' : 'negative'}
          icon={<CheckCircleIcon className="w-6 h-6" />}
          color="blue"
        />
        
        <StatsCard
          title="Average MTTR"
          value={`${Math.round(overallMetrics.averageMTTR)}m`}
          change={`Target: &lt;${slaTargets?.mttr_target_minutes || 30}m`}
          changeType={overallMetrics.averageMTTR <= (slaTargets?.mttr_target_minutes || 30) ? 'positive' : 'negative'}
          icon={<ClockIcon className="w-6 h-6" />}
          color="green"
        />
        
        <StatsCard
          title="Average MTBF"
          value={`${Math.round(overallMetrics.averageMTBF)}h`}
          change={`Target: >${slaTargets?.mtbf_target_hours || 720}h`}
          changeType={overallMetrics.averageMTBF >= (slaTargets?.mtbf_target_hours || 720) ? 'positive' : 'negative'}
          icon={<ArrowTrendingUpIcon className="w-6 h-6" />}
          color="indigo"
        />
        
        <StatsCard
          title="Total Incidents"
          value={overallMetrics.totalIncidents.toString()}
          change={`${complianceBreakdown.breach || 0} breaches`}
          changeType={complianceBreakdown.breach === 0 ? 'positive' : 'negative'}
          icon={<ExclamationTriangleIcon className="w-6 h-6" />}
          color="orange"
        />
        
        <StatsCard
          title="Compliance Rate"
          value={`${overallMetrics.complianceRate.toFixed(1)}%`}
          change={`${complianceBreakdown.compliant || 0}/${slaMetrics?.length || 0} sites`}
          changeType={overallMetrics.complianceRate >= 95 ? 'positive' : 'negative'}
          icon={<DocumentTextIcon className="w-6 h-6" />}
          color="purple"
        />
      </div>

      {/* Compliance Status Overview */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Compliance Status Overview</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div className="bg-green-50 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-green-600">
              {complianceBreakdown.compliant || 0}
            </div>
            <div className="text-sm text-green-600">Compliant Sites</div>
            <div className="text-xs text-gray-500 mt-1">
              Meeting all SLA targets
            </div>
          </div>
          
          <div className="bg-yellow-50 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-yellow-600">
              {complianceBreakdown.at_risk || 0}
            </div>
            <div className="text-sm text-yellow-600">At Risk Sites</div>
            <div className="text-xs text-gray-500 mt-1">
              Close to SLA thresholds
            </div>
          </div>
          
          <div className="bg-red-50 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-red-600">
              {complianceBreakdown.breach || 0}
            </div>
            <div className="text-sm text-red-600">SLA Breaches</div>
            <div className="text-xs text-gray-500 mt-1">
              Not meeting targets
            </div>
          </div>
        </div>
      </div>

      {/* Main SLA Dashboard Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* SLA Trends Chart */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-lg font-semibold text-gray-900">SLA Trends</h2>
            <div className="flex space-x-2">
              <div className="flex items-center space-x-1">
                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                <span className="text-sm text-gray-600">Availability</span>
              </div>
              <div className="flex items-center space-x-1">
                <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                <span className="text-sm text-gray-600">MTTR</span>
              </div>
              <div className="flex items-center space-x-1">
                <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
                <span className="text-sm text-gray-600">MTBF</span>
              </div>
            </div>
          </div>
          <SLAChart data={slaMetrics} period={selectedPeriod} targets={slaTargets} />
        </div>

        {/* Recent Incidents Timeline */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Recent SLA Incidents</h2>
          <IncidentTimeline 
            incidents={slaIncidents?.slice(0, 10) || []} 
            showFilters={false}
            compact={true}
          />
          
          {slaIncidents && slaIncidents.length > 10 && (
            <div className="mt-4 text-center">
              <a
                href="/incidents"
                className="text-sm text-blue-600 hover:text-blue-800 font-medium"
              >
                View all incidents ({slaIncidents.length})
              </a>
            </div>
          )}
        </div>
      </div>

      {/* Detailed SLA Table */}
      <div className="bg-white rounded-lg shadow-md">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900">Site-by-Site SLA Performance</h2>
        </div>
        <SLATable 
          metrics={slaMetrics || []} 
          targets={slaTargets}
          onSiteSelect={(siteIds) => setSelectedSites(siteIds)}
          selectedSites={selectedSites}
        />
      </div>

      {/* SLA Target Editor Modal */}
      {showTargetEditor && (
        <SLATargetEditor
          targets={slaTargets}
          onClose={() => setShowTargetEditor(false)}
          onSave={() => {
            setShowTargetEditor(false);
            // Refetch targets after save
            // queryClient.invalidateQueries(['sla-targets']);
          }}
        />
      )}
    </div>
  );
};

export default SLAPage;