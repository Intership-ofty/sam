import React, { useMemo } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface SLAMetric {
  metric_name: string;
  availability_percentage: number;
  mttr_minutes: number;
  mtbf_hours: number;
  period_start: string;
  period_end: string;
}

interface SLATargets {
  availability_target: number;
  mttr_target_minutes: number;
  mtbf_target_hours: number;
}

interface SLAChartProps {
  data: SLAMetric[];
  period: string;
  targets?: SLATargets;
  height?: number;
}

const SLAChart: React.FC<SLAChartProps> = ({ 
  data, 
  period, 
  targets,
  height = 300 
}) => {
  const chartData = useMemo(() => {
    if (!data?.length) {
      return {
        labels: [],
        datasets: []
      };
    }

    // Group data by time periods
    const timeLabels = data.map((item, index) => {
      if (period === 'current_month' || period === 'last_month') {
        return `Week ${Math.floor(index / 7) + 1}`;
      } else if (period === 'current_quarter' || period === 'last_quarter') {
        return `Month ${(index % 3) + 1}`;
      } else {
        return new Date(item.period_start).toLocaleDateString();
      }
    });

    // Calculate averages for each time period
    const availabilityData = data.map(item => item.availability_percentage);
    const mttrData = data.map(item => item.mttr_minutes);
    const mtbfData = data.map(item => item.mtbf_hours / 24); // Convert to days for better visualization

    return {
      labels: [...new Set(timeLabels)], // Remove duplicates
      datasets: [
        {
          label: 'Availability %',
          data: availabilityData,
          borderColor: 'rgb(34, 197, 94)',
          backgroundColor: 'rgba(34, 197, 94, 0.1)',
          fill: true,
          yAxisID: 'y',
          tension: 0.4,
        },
        {
          label: 'MTTR (minutes)',
          data: mttrData,
          borderColor: 'rgb(59, 130, 246)',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          fill: false,
          yAxisID: 'y1',
          tension: 0.4,
        },
        {
          label: 'MTBF (days)',
          data: mtbfData,
          borderColor: 'rgb(147, 51, 234)',
          backgroundColor: 'rgba(147, 51, 234, 0.1)',
          fill: false,
          yAxisID: 'y2',
          tension: 0.4,
        },
        // Target lines
        ...(targets ? [
          {
            label: 'Availability Target',
            data: Array(availabilityData.length).fill(targets.availability_target),
            borderColor: 'rgba(34, 197, 94, 0.5)',
            backgroundColor: 'transparent',
            borderDash: [5, 5],
            pointRadius: 0,
            yAxisID: 'y',
            tension: 0,
          },
          {
            label: 'MTTR Target',
            data: Array(mttrData.length).fill(targets.mttr_target_minutes),
            borderColor: 'rgba(59, 130, 246, 0.5)',
            backgroundColor: 'transparent',
            borderDash: [5, 5],
            pointRadius: 0,
            yAxisID: 'y1',
            tension: 0,
          }
        ] : [])
      ]
    };
  }, [data, period, targets]);

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: false,
      },
      tooltip: {
        mode: 'index' as const,
        intersect: false,
        callbacks: {
          label: function(context: any) {
            let label = context.dataset.label || '';
            if (label) {
              label += ': ';
            }
            if (context.parsed.y !== null) {
              if (label.includes('Availability')) {
                label += context.parsed.y.toFixed(3) + '%';
              } else if (label.includes('MTTR')) {
                label += context.parsed.y.toFixed(1) + ' minutes';
              } else if (label.includes('MTBF')) {
                label += context.parsed.y.toFixed(1) + ' days';
              }
            }
            return label;
          }
        }
      },
    },
    interaction: {
      mode: 'nearest' as const,
      axis: 'x' as const,
      intersect: false,
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Time Period'
        }
      },
      y: {
        type: 'linear' as const,
        display: true,
        position: 'left' as const,
        title: {
          display: true,
          text: 'Availability (%)',
          color: 'rgb(34, 197, 94)',
        },
        min: Math.min(99, Math.min(...chartData.datasets[0]?.data || [99]) - 0.1),
        max: 100,
        ticks: {
          color: 'rgb(34, 197, 94)',
          callback: function(value: any) {
            return value.toFixed(2) + '%';
          }
        }
      },
      y1: {
        type: 'linear' as const,
        display: true,
        position: 'right' as const,
        title: {
          display: true,
          text: 'MTTR (minutes)',
          color: 'rgb(59, 130, 246)',
        },
        ticks: {
          color: 'rgb(59, 130, 246)',
        },
        grid: {
          drawOnChartArea: false,
        },
      },
      y2: {
        type: 'linear' as const,
        display: false,
        position: 'right' as const,
        title: {
          display: true,
          text: 'MTBF (days)',
          color: 'rgb(147, 51, 234)',
        },
        ticks: {
          color: 'rgb(147, 51, 234)',
        },
        grid: {
          drawOnChartArea: false,
        },
      },
    },
  };

  if (!data?.length) {
    return (
      <div 
        className="flex items-center justify-center bg-gray-50 rounded-lg"
        style={{ height: `${height}px` }}
      >
        <div className="text-center">
          <div className="text-gray-400 text-lg mb-2">No SLA data available</div>
          <div className="text-gray-500 text-sm">
            SLA metrics will appear here once data is collected
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={{ height: `${height}px` }}>
      <Line data={chartData} options={options} />
    </div>
  );
};

export default SLAChart;