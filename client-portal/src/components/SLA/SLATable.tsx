import React, { useState, useMemo } from 'react';
import {
  CheckCircleIcon,
  ExclamationTriangleIcon,
  XCircleIcon,
  ClockIcon,
  ArrowTrendingUpIcon,
  MagnifyingGlassIcon,
  ChevronUpDownIcon,
  ChevronDownIcon,
  ChevronUpIcon,
} from '@heroicons/react/24/outline';

interface SLAMetric {
  metric_name: string;
  site_id?: string;
  site_name?: string;
  target_percentage: number;
  current_percentage: number;
  availability_percentage: number;
  mttr_minutes: number;
  mtbf_hours: number;
  incidents_count: number;
  downtime_minutes: number;
  compliance_status: 'compliant' | 'at_risk' | 'breach';
  period_start: string;
  period_end: string;
}

interface SLATargets {
  availability_target: number;
  mttr_target_minutes: number;
  mtbf_target_hours: number;
}

interface SLATableProps {
  metrics: SLAMetric[];
  targets?: SLATargets;
  onSiteSelect?: (siteIds: string[]) => void;
  selectedSites: string[];
}

type SortField = 'site_name' | 'availability_percentage' | 'mttr_minutes' | 'mtbf_hours' | 'incidents_count' | 'compliance_status';
type SortDirection = 'asc' | 'desc';

const SLATable: React.FC<SLATableProps> = ({ 
  metrics, 
  targets, 
  onSiteSelect,
  selectedSites 
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [sortField, setSortField] = useState<SortField>('compliance_status');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');
  const [filterStatus, setFilterStatus] = useState<string>('all');

  // Filter and sort data
  const filteredAndSortedData = useMemo(() => {
    let filtered = metrics.filter(metric => {
      const matchesSearch = (metric.site_name || metric.metric_name)
        .toLowerCase()
        .includes(searchTerm.toLowerCase());
      
      const matchesFilter = filterStatus === 'all' || metric.compliance_status === filterStatus;
      
      return matchesSearch && matchesFilter;
    });

    // Sort data
    filtered.sort((a, b) => {
      let aVal, bVal;
      
      switch (sortField) {
        case 'site_name':
          aVal = a.site_name || a.metric_name;
          bVal = b.site_name || b.metric_name;
          break;
        case 'availability_percentage':
          aVal = a.availability_percentage;
          bVal = b.availability_percentage;
          break;
        case 'mttr_minutes':
          aVal = a.mttr_minutes;
          bVal = b.mttr_minutes;
          break;
        case 'mtbf_hours':
          aVal = a.mtbf_hours;
          bVal = b.mtbf_hours;
          break;
        case 'incidents_count':
          aVal = a.incidents_count;
          bVal = b.incidents_count;
          break;
        case 'compliance_status':
          const statusOrder = { 'breach': 3, 'at_risk': 2, 'compliant': 1 };
          aVal = statusOrder[a.compliance_status];
          bVal = statusOrder[b.compliance_status];
          break;
        default:
          return 0;
      }
      
      if (typeof aVal === 'string' && typeof bVal === 'string') {
        const comparison = aVal.localeCompare(bVal);
        return sortDirection === 'asc' ? comparison : -comparison;
      }
      
      const comparison = (aVal as number) - (bVal as number);
      return sortDirection === 'asc' ? comparison : -comparison;
    });

    return filtered;
  }, [metrics, searchTerm, sortField, sortDirection, filterStatus]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  };

  const handleSiteSelection = (siteId: string) => {
    if (!onSiteSelect) return;
    
    const newSelection = selectedSites.includes(siteId)
      ? selectedSites.filter(id => id !== siteId)
      : [...selectedSites, siteId];
    
    onSiteSelect(newSelection);
  };

  const getComplianceIcon = (status: string) => {
    switch (status) {
      case 'compliant':
        return <CheckCircleIcon className="w-5 h-5 text-green-500" />;
      case 'at_risk':
        return <ExclamationTriangleIcon className="w-5 h-5 text-yellow-500" />;
      case 'breach':
        return <XCircleIcon className="w-5 h-5 text-red-500" />;
      default:
        return null;
    }
  };

  const getComplianceBadge = (status: string) => {
    const baseClasses = "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium";
    
    switch (status) {
      case 'compliant':
        return `${baseClasses} bg-green-100 text-green-800`;
      case 'at_risk':
        return `${baseClasses} bg-yellow-100 text-yellow-800`;
      case 'breach':
        return `${baseClasses} bg-red-100 text-red-800`;
      default:
        return `${baseClasses} bg-gray-100 text-gray-800`;
    }
  };

  const getSortIcon = (field: SortField) => {
    if (sortField !== field) {
      return <ChevronUpDownIcon className="w-4 h-4 text-gray-400" />;
    }
    
    return sortDirection === 'asc' 
      ? <ChevronUpIcon className="w-4 h-4 text-blue-500" />
      : <ChevronDownIcon className="w-4 h-4 text-blue-500" />;
  };

  const formatDuration = (minutes: number): string => {
    if (minutes < 60) return `${minutes.toFixed(0)}m`;
    const hours = Math.floor(minutes / 60);
    const remainingMinutes = Math.round(minutes % 60);
    return remainingMinutes > 0 ? `${hours}h ${remainingMinutes}m` : `${hours}h`;
  };

  const formatMTBF = (hours: number): string => {
    if (hours < 24) return `${hours.toFixed(1)}h`;
    const days = Math.floor(hours / 24);
    const remainingHours = Math.round(hours % 24);
    return remainingHours > 0 ? `${days}d ${remainingHours}h` : `${days}d`;
  };

  return (
    <div className="space-y-4">
      {/* Search and Filter Controls */}
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="relative flex-1">
          <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search sites..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10 pr-4 py-2 w-full border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        
        <select
          value={filterStatus}
          onChange={(e) => setFilterStatus(e.target.value)}
          className="px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="all">All Status</option>
          <option value="compliant">Compliant</option>
          <option value="at_risk">At Risk</option>
          <option value="breach">SLA Breach</option>
        </select>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full bg-white rounded-lg overflow-hidden">
          <thead className="bg-gray-50">
            <tr>
              {onSiteSelect && (
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Select
                </th>
              )}
              <th 
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('site_name')}
              >
                <div className="flex items-center space-x-1">
                  <span>Site</span>
                  {getSortIcon('site_name')}
                </div>
              </th>
              <th 
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('compliance_status')}
              >
                <div className="flex items-center space-x-1">
                  <span>Status</span>
                  {getSortIcon('compliance_status')}
                </div>
              </th>
              <th 
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('availability_percentage')}
              >
                <div className="flex items-center space-x-1">
                  <span>Availability</span>
                  {getSortIcon('availability_percentage')}
                </div>
              </th>
              <th 
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('mttr_minutes')}
              >
                <div className="flex items-center space-x-1">
                  <span>MTTR</span>
                  {getSortIcon('mttr_minutes')}
                </div>
              </th>
              <th 
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('mtbf_hours')}
              >
                <div className="flex items-center space-x-1">
                  <span>MTBF</span>
                  {getSortIcon('mtbf_hours')}
                </div>
              </th>
              <th 
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('incidents_count')}
              >
                <div className="flex items-center space-x-1">
                  <span>Incidents</span>
                  {getSortIcon('incidents_count')}
                </div>
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Downtime
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200">
            {filteredAndSortedData.map((metric, index) => {
              const siteId = metric.site_id || metric.metric_name;
              const siteName = metric.site_name || metric.metric_name;
              const isSelected = selectedSites.includes(siteId);
              
              return (
                <tr 
                  key={siteId}
                  className={`hover:bg-gray-50 ${isSelected ? 'bg-blue-50' : ''}`}
                >
                  {onSiteSelect && (
                    <td className="px-6 py-4 whitespace-nowrap">
                      <input
                        type="checkbox"
                        checked={isSelected}
                        onChange={() => handleSiteSelection(siteId)}
                        className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                      />
                    </td>
                  )}
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm font-medium text-gray-900">
                      {siteName}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center space-x-2">
                      {getComplianceIcon(metric.compliance_status)}
                      <span className={getComplianceBadge(metric.compliance_status)}>
                        {metric.compliance_status.charAt(0).toUpperCase() + metric.compliance_status.slice(1)}
                      </span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center space-x-2">
                      <div className="text-sm font-medium text-gray-900">
                        {metric.availability_percentage.toFixed(3)}%
                      </div>
                      {targets && (
                        <div className={`text-xs ${
                          metric.availability_percentage >= targets.availability_target 
                            ? 'text-green-600' 
                            : 'text-red-600'
                        }`}>
                          Target: {targets.availability_target}%
                        </div>
                      )}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center space-x-2">
                      <ClockIcon className="w-4 h-4 text-gray-400" />
                      <div className="text-sm text-gray-900">
                        {formatDuration(metric.mttr_minutes)}
                      </div>
                      {targets && (
                        <div className={`text-xs ${
                          metric.mttr_minutes <= targets.mttr_target_minutes 
                            ? 'text-green-600' 
                            : 'text-red-600'
                        }`}>
                          Target: &lt; {formatDuration(targets.mttr_target_minutes)}
                        </div>
                      )}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center space-x-2">
                      <ArrowTrendingUpIcon className="w-4 h-4 text-gray-400" />
                      <div className="text-sm text-gray-900">
                        {formatMTBF(metric.mtbf_hours)}
                      </div>
                      {targets && (
                        <div className={`text-xs ${
                          metric.mtbf_hours >= targets.mtbf_target_hours 
                            ? 'text-green-600' 
                            : 'text-red-600'
                        }`}>
                          Target: >{formatMTBF(targets.mtbf_target_hours)}
                        </div>
                      )}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className={`text-sm font-medium ${
                      metric.incidents_count === 0 ? 'text-green-600' : 
                      metric.incidents_count < 3 ? 'text-yellow-600' : 'text-red-600'
                    }`}>
                      {metric.incidents_count}
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm text-gray-900">
                      {formatDuration(metric.downtime_minutes)}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        
        {filteredAndSortedData.length === 0 && (
          <div className="text-center py-12">
            <div className="text-gray-500 text-lg">No SLA data found</div>
            <div className="text-gray-400 text-sm mt-2">
              {searchTerm ? 'Try adjusting your search criteria' : 'SLA metrics will appear here once available'}
            </div>
          </div>
        )}
      </div>

      {/* Summary Footer */}
      {filteredAndSortedData.length > 0 && (
        <div className="bg-gray-50 px-6 py-3 rounded-lg">
          <div className="flex justify-between items-center text-sm text-gray-600">
            <div>
              Showing {filteredAndSortedData.length} of {metrics.length} sites
            </div>
            <div className="flex space-x-6">
              <div>
                Avg Availability: <span className="font-medium">
                  {(filteredAndSortedData.reduce((sum, m) => sum + m.availability_percentage, 0) / filteredAndSortedData.length).toFixed(3)}%
                </span>
              </div>
              <div>
                Avg MTTR: <span className="font-medium">
                  {formatDuration(filteredAndSortedData.reduce((sum, m) => sum + m.mttr_minutes, 0) / filteredAndSortedData.length)}
                </span>
              </div>
              <div>
                Total Incidents: <span className="font-medium">
                  {filteredAndSortedData.reduce((sum, m) => sum + m.incidents_count, 0)}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SLATable;