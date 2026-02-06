import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LabelList, PieChart, Pie, Cell } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Loader2 } from 'lucide-react';

const CRITERIA_MAPPING = {
    "The Teaching-Learning Environment": "Q1",
    "System of Monitoring Students Progress": "Q2",
    "Involvement of Faculty in Teaching-Learning": "Q3",
    "Infrastructure Facilities": "Q4",
    "Learning Resources Like Library, Internet, Computing etc.": "Q5",
    "Study Environment and Discipline": "Q6",
    "Counselling and Placements": "Q7",
    "Institute’s overall Support Facilities": "Q8",
    "Parental Perception about Institute": "Q9",
    "Students overall Holistic Development": "Q10",
};

const SHORT_LABELS = Object.values(CRITERIA_MAPPING);

export function AnalyticsView() {
    const [department, setDepartment] = useState('CSE');
    const [className, setClassName] = useState('');

    // Data states
    const [distData, setDistData] = useState(null); // Stacked Bar
    const [overallData, setOverallData] = useState(null); // Pie
    const [averageData, setAverageData] = useState(null); // Bar (Avg)

    const [loading, setLoading] = useState(false);

    const COLORS = {
        '1': '#ef4444', // Poor
        '2': '#eab308', // Average
        '3': '#3b82f6', // Good
        '4': '#22c55e'  // Excellent
    };

    const fetchData = async () => {
        setLoading(true);
        try {
            let url = `/analytics/department/criteria?dept=${department}`;
            if (className) {
                url += `&cls=${className}`;
            }
            const response = await fetch(url);
            const json = await response.json();

            // 1. Criterion-wise Distribution (Stacked Bar)
            const transformedDist = Object.entries(json.criteria).map(([criterion, ratings]) => {
                return {
                    name: CRITERIA_MAPPING[criterion] || criterion.substring(0, 10),
                    full_name: criterion,
                    ...ratings
                };
            });
            // Sort Q1 -> Q10
            transformedDist.sort((a, b) => {
                const numA = parseInt(a.name.replace('Q', ''));
                const numB = parseInt(b.name.replace('Q', ''));
                return numA - numB;
            });
            setDistData(transformedDist);

            // 2. Overall Satisfaction (Pie Chart)
            // Aggegate counts across all questions
            const totals = { '1': 0, '2': 0, '3': 0, '4': 0 };
            Object.values(json.criteria).forEach(ratings => {
                totals['1'] += ratings['1'] || 0;
                totals['2'] += ratings['2'] || 0;
                totals['3'] += ratings['3'] || 0;
                totals['4'] += ratings['4'] || 0;
            });

            const transformedOverall = [
                { name: 'Excellent (4)', value: totals['4'], color: COLORS['4'] },
                { name: 'Good (3)', value: totals['3'], color: COLORS['3'] },
                { name: 'Average (2)', value: totals['2'], color: COLORS['2'] },
                { name: 'Poor (1)', value: totals['1'], color: COLORS['1'] },
            ].filter(d => d.value > 0); // Hide empty slices
            setOverallData(transformedOverall);

            // 3. Average Score per Criterion (Bar Chart)
            const transformedAvg = Object.entries(json.criteria).map(([criterion, ratings]) => {
                const c1 = ratings['1'] || 0;
                const c2 = ratings['2'] || 0;
                const c3 = ratings['3'] || 0;
                const c4 = ratings['4'] || 0;
                const total = c1 + c2 + c3 + c4;

                const weightedSum = (1 * c1) + (2 * c2) + (3 * c3) + (4 * c4);
                const avg = total > 0 ? (weightedSum / total).toFixed(2) : 0;

                return {
                    name: CRITERIA_MAPPING[criterion] || criterion.substring(0, 5),
                    full_name: criterion,
                    avg: parseFloat(avg)
                };
            });
            // Sort by Best (Highest Avg) to Worst
            transformedAvg.sort((a, b) => b.avg - a.avg);
            setAverageData(transformedAvg);

        } catch (err) {
            console.error("Failed to fetch analytics:", err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchData();
    }, [department]);

    const handleApply = () => {
        fetchData();
    }

    return (
        <div className="space-y-12 animate-in fade-in duration-500 pb-20">

            {/* Controls */}
            <div className="flex flex-col md:flex-row gap-4 items-end bg-slate-50 p-6 rounded-xl border border-slate-100 shadow-sm">
                <div className="space-y-1 flex-1 w-full">
                    <Label className="font-bold text-slate-700">Department</Label>
                    <Input
                        value={department}
                        onChange={(e) => setDepartment(e.target.value)}
                        placeholder="CSE"
                        className="bg-white"
                    />
                </div>
                <div className="space-y-1 flex-1 w-full">
                    <Label className="font-bold text-slate-700">Class (Optional)</Label>
                    <Input
                        value={className}
                        onChange={(e) => setClassName(e.target.value)}
                        placeholder="e.g. SE-A"
                        className="bg-white"
                    />
                </div>
                <Button onClick={handleApply} disabled={loading} className="min-w-[140px] shadow-sm">
                    {loading ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : "Refresh Data"}
                </Button>
            </div>

            {distData ? (
                <>
                    {/* SECTION 1: Criterion-wise Rating Distribution */}


                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">

                        {/* SECTION 2: Overall Satisfaction */}
                        <div className="space-y-4">
                            <div className="space-y-1">
                                <h2 className="text-xl font-serif font-bold text-slate-800">Overall Satisfaction Distribution</h2>
                            </div>
                            <Card className="border-none shadow-xl shadow-slate-200/50">
                                <CardContent className="h-[400px] pt-6">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <PieChart>
                                            <Pie
                                                data={overallData}
                                                cx="50%"
                                                cy="50%"
                                                labelLine={false}
                                                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                                                outerRadius={120}
                                                fill="#8884d8"
                                                dataKey="value"
                                            >
                                                {overallData.map((entry, index) => (
                                                    <Cell key={`cell-${index}`} fill={entry.color} />
                                                ))}
                                            </Pie>
                                            <Tooltip />
                                            <Legend verticalAlign="bottom" height={36} />
                                        </PieChart>
                                    </ResponsiveContainer>
                                </CardContent>
                            </Card>
                        </div>

                        {/* SECTION 3: Average Score */}
                        <div className="space-y-4">
                            <div className="space-y-1">
                                <h2 className="text-xl font-serif font-bold text-slate-800">Average Score per Criterion</h2>
                            </div>
                            <Card className="border-none shadow-xl shadow-slate-200/50">
                                <CardContent className="h-[400px] pt-6">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <BarChart
                                            data={averageData}
                                            layout="vertical"
                                            margin={{ top: 20, right: 30, left: 40, bottom: 5 }}
                                        >
                                            <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={true} />
                                            <XAxis type="number" domain={[0, 4]} />
                                            <YAxis dataKey="name" type="category" width={40} tick={{ fontSize: 12, fontWeight: 'bold' }} />
                                            <Tooltip
                                                cursor={{ fill: 'rgba(0,0,0,0.05)' }}
                                                formatter={(value) => [value, "Avg Score"]}
                                            />
                                            <Bar dataKey="avg" name="Average Score" fill="#0f172a" radius={[0, 4, 4, 0]}>
                                                <LabelList dataKey="avg" position="right" className="font-bold text-xs" />
                                            </Bar>
                                        </BarChart>
                                    </ResponsiveContainer>
                                </CardContent>
                            </Card>
                        </div>

                    </div>

                </>
            ) : (
                <div className="h-64 flex flex-col items-center justify-center text-muted-foreground bg-slate-50 rounded-xl border border-dashed border-slate-200">
                    <p>Enter details and click refresh to visualize data</p>
                </div>
            )}

            {/* Legend / Key */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm text-slate-600 bg-slate-50 p-6 rounded-xl border border-slate-100">
                {Object.entries(CRITERIA_MAPPING).map(([full, short]) => (
                    <div key={short} className="flex gap-3">
                        <span className="font-bold min-w-[30px] bg-white px-2 py-0.5 rounded border border-slate-200 text-center text-xs flex items-center justify-center h-6">{short}</span>
                        <span className="text-xs flex items-center">{full}</span>
                    </div>
                ))}
            </div>

        </div>
    );
}
