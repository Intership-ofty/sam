import React, { useState } from 'react';
import {
  ExclamationTriangleIcon,
  XCircleIcon,
  ClockIcon,
  CheckCircleIcon,
  CalendarIcon,
  MapPinIcon,
  UserIcon,
} from '@heroicons/react/24/outline';

interface SLAIncident {
  id: string;
  site_id: string;
  site_name: string;
  title: string;
  description: string;
  severity: 'critical' | 'major' | 'minor' | 'warning';
  status: 'open' | 'investigating' | 'resolving' | 'resolved' | 'closed';
  started_at: string;
  resolved_at?: string;
  closed_at?: string;
  duration_minutes?: number;
  sla_breach: boolean;
  assignee?: string;
  impact_description: string;
  resolution_summary?: string;
  root_cause?: string;
}

interface IncidentTimelineProps {
  incidents: SLAIncident[];
  showFilters?: boolean;
  compact?: boolean;
  limit?: number;
}

const IncidentTimeline: React.FC<IncidentTimelineProps> = ({ 
  incidents, 
  showFilters = true,
  compact = false,
  limit 
}) => {
  const [severityFilter, setSeverityFilter] = useState<string>('all');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [slaFilter, setSlaFilter] = useState<string>('all');

  // Filter incidents
  const filteredIncidents = incidents
    .filter(incident => {
      const matchesSeverity = severityFilter === 'all' || incident.severity === severityFilter;
      const matchesStatus = statusFilter === 'all' || incident.status === statusFilter;
      const matchesSLA = slaFilter === 'all' || 
        (slaFilter === 'breach' && incident.sla_breach) ||
        (slaFilter === 'no_breach' && !incident.sla_breach);
      
      return matchesSeverity && matchesStatus && matchesSLA;
    })
    .slice(0, limit);

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical':
        return <XCircleIcon className="w-5 h-5 text-red-500" />;
      case 'major':
        return <ExclamationTriangleIcon className="w-5 h-5 text-orange-500" />;
      case 'minor':
        return <ExclamationTriangleIcon className="w-5 h-5 text-yellow-500" />;
      case 'warning':
        return <ExclamationTriangleIcon className="w-5 h-5 text-blue-500" />;
      default:
        return <ExclamationTriangleIcon className="w-5 h-5 text-gray-500" />;
    }
  };

  const getSeverityBadge = (severity: string) => {
    const baseClasses = "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium";
    
    switch (severity) {
      case 'critical':
        return `${baseClasses} bg-red-100 text-red-800`;
      case 'major':
        return `${baseClasses} bg-orange-100 text-orange-800`;
      case 'minor':
        return `${baseClasses} bg-yellow-100 text-yellow-800`;
      case 'warning':
        return `${baseClasses} bg-blue-100 text-blue-800`;
      default:
        return `${baseClasses} bg-gray-100 text-gray-800`;
    }
  };

  const getStatusBadge = (status: string) => {
    const baseClasses = "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium";
    
    switch (status) {
      case 'open':
        return `${baseClasses} bg-red-100 text-red-800`;
      case 'investigating':
        return `${baseClasses} bg-yellow-100 text-yellow-800`;
      case 'resolving':
        return `${baseClasses} bg-blue-100 text-blue-800`;
      case 'resolved':
        return `${baseClasses} bg-green-100 text-green-800`;
      case 'closed':
        return `${baseClasses} bg-gray-100 text-gray-800`;
      default:
        return `${baseClasses} bg-gray-100 text-gray-800`;
    }
  };

  const formatDuration = (minutes?: number): string => {
    if (!minutes) return 'Ongoing';
    
    if (minutes < 60) return `${minutes}m`;
    const hours = Math.floor(minutes / 60);
    const remainingMinutes = Math.round(minutes % 60);
    return remainingMinutes > 0 ? `${hours}h ${remainingMinutes}m` : `${hours}h`;
  };

  const formatTimestamp = (timestamp: string): string => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffHours / 24);

    if (diffHours < 1) {
      const diffMinutes = Math.floor(diffMs / (1000 * 60));
      return `${diffMinutes}m ago`;
    } else if (diffHours < 24) {
      return `${diffHours}h ago`;
    } else if (diffDays < 7) {
      return `${diffDays}d ago`;
    } else {
      return date.toLocaleDateString();
    }
  };

  if (!incidents.length) {
    return (
      <div className="text-center py-8">
        <CheckCircleIcon className="w-12 h-12 text-green-500 mx-auto mb-3" />
        <div className="text-gray-500 text-lg">No incidents found</div>
        <div className="text-gray-400 text-sm">
          All systems are operating normally
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Filters */}
      {showFilters && (
        <div className="flex flex-wrap gap-3">
          <select
            value={severityFilter}
            onChange={(e) => setSeverityFilter(e.target.value)}
            className="px-3 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">All Severities</option>
            <option value="critical">Critical</option>
            <option value="major">Major</option>
            <option value="minor">Minor</option>
            <option value="warning">Warning</option>
          </select>
          
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="px-3 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">All Status</option>
            <option value="open">Open</option>
            <option value="investigating">Investigating</option>
            <option value="resolving">Resolving</option>
            <option value="resolved">Resolved</option>
            <option value="closed">Closed</option>
          </select>
          
          <select
            value={slaFilter}
            onChange={(e) => setSlaFilter(e.target.value)}
            className="px-3 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">All SLA</option>
            <option value="breach">SLA Breach</option>
            <option value="no_breach">No Breach</option>
          </select>
        </div>
      )}

      {/* Timeline */}
      <div className="space-y-3">
        {filteredIncidents.map((incident, index) => (
          <div
            key={incident.id}
            className={`relative ${
              compact ? 'pl-6' : 'pl-10'
            } pb-4 ${
              index !== filteredIncidents.length - 1 ? 'border-l-2 border-gray-200' : ''
            }`}
          >
            {/* Timeline dot */}
            <div className={`absolute ${
              compact ? '-left-2' : '-left-3'
            } top-2 ${
              compact ? 'w-4 h-4' : 'w-6 h-6'
            } rounded-full bg-white border-2 ${
              incident.sla_breach ? 'border-red-500' : 'border-blue-500'
            } flex items-center justify-center`}>
              {compact ? (
                <div className={`w-2 h-2 rounded-full ${
                  incident.sla_breach ? 'bg-red-500' : 'bg-blue-500'
                }`} />
              ) : (
                <div className="text-xs font-bold text-gray-600">
                  {index + 1}
                </div>
              )}
            </div>

            {/* Incident card */}
            <div className={`bg-white rounded-lg border shadow-sm p-4 ${
              incident.sla_breach ? 'border-l-4 border-l-red-500' : ''
            }`}>
              {/* Header */}
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center space-x-2">
                  {getSeverityIcon(incident.severity)}
                  <h3 className={`font-medium ${
                    compact ? 'text-sm' : 'text-base'
                  } text-gray-900`}>
                    {incident.title}
                  </h3>
                  {incident.sla_breach && (
                    <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800">
                      SLA Breach
                    </span>
                  )}
                </div>
                
                <div className="flex items-center space-x-2">
                  <span className={getSeverityBadge(incident.severity)}>
                    {incident.severity.charAt(0).toUpperCase() + incident.severity.slice(1)}
                  </span>
                  <span className={getStatusBadge(incident.status)}>
                    {incident.status.charAt(0).toUpperCase() + incident.status.slice(1)}
                  </span>
                </div>
              </div>

              {/* Details */}
              {!compact && (
                <div className="mb-3">
                  <p className="text-sm text-gray-600">
                    {incident.description}
                  </p>
                  {incident.impact_description && (
                    <p className="text-sm text-red-600 mt-1">
                      Impact: {incident.impact_description}
                    </p>
                  )}
                </div>
              )}

              {/* Metadata */}
              <div className="flex flex-wrap items-center gap-4 text-xs text-gray-500">
                <div className="flex items-center space-x-1">
                  <MapPinIcon className="w-3 h-3" />
                  <span>{incident.site_name}</span>
                </div>
                
                <div className="flex items-center space-x-1">
                  <CalendarIcon className="w-3 h-3" />
                  <span>{formatTimestamp(incident.started_at)}</span>
                </div>
                
                <div className="flex items-center space-x-1">
                  <ClockIcon className="w-3 h-3" />
                  <span>{formatDuration(incident.duration_minutes)}</span>
                </div>
                
                {incident.assignee && (
                  <div className="flex items-center space-x-1">
                    <UserIcon className="w-3 h-3" />
                    <span>{incident.assignee}</span>
                  </div>
                )}
              </div>

              {/* Resolution details */}
              {!compact && incident.resolution_summary && (
                <div className="mt-3 pt-3 border-t border-gray-100">
                  <p className="text-sm text-gray-700">
                    <span className="font-medium">Resolution:</span> {incident.resolution_summary}
                  </p>
                  {incident.root_cause && (
                    <p className="text-sm text-gray-600 mt-1">
                      <span className="font-medium">Root Cause:</span> {incident.root_cause}
                    </p>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Show more link */}
      {limit && incidents.length > limit && (
        <div className="text-center pt-4">
          <a
            href="/incidents"
            className="text-sm text-blue-600 hover:text-blue-800 font-medium"
          >
            View all {incidents.length} incidents →
          </a>
        </div>
      )}
    </div>
  );
};

export default IncidentTimeline;