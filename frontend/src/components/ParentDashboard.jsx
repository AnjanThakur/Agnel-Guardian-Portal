import React, { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LabelList, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Loader2, Search, User, GraduationCap, Download } from 'lucide-react';

export function ParentDashboard() {
    // Simulating a logged-in parent who has two children in the college
    const siblings = ['S001', 'S002'];
    const [activeSibling, setActiveSibling] = useState(siblings[0]);

    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const [studentData, setStudentData] = useState(null);
    const [selectedSemester, setSelectedSemester] = useState(null);

    useEffect(() => {
        const fetchStudentData = async () => {
            if (!activeSibling) return;
            setLoading(true);
            setError('');

            try {
                const response = await fetch(`http://127.0.0.1:8000/student/${activeSibling}`);
                if (!response.ok) {
                    if (response.status === 404) {
                        throw new Error(`Student ${activeSibling} not found in database.`);
                    }
                    throw new Error("Failed to fetch student data.");
                }
                const data = await response.json();
                setStudentData(data);

                // Default to latest semester available for bar chart
                if (data.semesters.length > 0) {
                    const maxSem = Math.max(...data.semesters.map(s => s.semester));
                    setSelectedSemester(maxSem);
                }
            } catch (err) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        fetchStudentData();
    }, [activeSibling]);

    // Prepare line chart data
    const progressionData = studentData?.semesters.map(s => ({
        semester: `Sem ${s.semester}`,
        percentage: s.percentage,
        sgpa: s.sgpa
    })) || [];

    // Prepare bar/radar data for selected semester
    const currentSemData = studentData?.semesters.find(s => s.semester === selectedSemester);
    const subjectData = currentSemData?.marks.map(m => ({
        subject: m.subject,
        obtained: m.marks_obtained,
        max: m.max_marks,
        percentage: (m.marks_obtained / m.max_marks) * 100
    })) || [];

    return (
        <div className="space-y-8 animate-in fade-in duration-500 pb-20 max-w-6xl mx-auto">

            {/* Header */}
            <div className="text-center space-y-4 max-w-2xl mx-auto mb-6">
                <h1 className="text-4xl font-serif font-medium text-primary tracking-tight">
                    Welcome to Parent Portal
                </h1>
                <p className="text-muted-foreground text-lg font-light">
                    Review your wards' comprehensive academic progress.
                </p>
            </div>

            {/* Sibling Switcher */}
            <div className="flex justify-center items-center gap-4 mb-8">
                <div className="bg-slate-100 p-1.5 rounded-xl inline-flex shadow-sm">
                    {siblings.map((sib) => (
                        <button
                            key={sib}
                            onClick={() => setActiveSibling(sib)}
                            className={`px-6 py-2.5 rounded-lg font-medium text-sm transition-all duration-200 ${activeSibling === sib
                                    ? 'bg-white text-primary shadow-md transform scale-105'
                                    : 'text-slate-500 hover:text-slate-800 hover:bg-slate-200/50'
                                }`}
                        >
                            <span className="flex items-center gap-2">
                                <User className="w-4 h-4" /> Student {sib}
                            </span>
                        </button>
                    ))}
                </div>
            </div>

            {loading && (
                <div className="flex justify-center items-center py-20">
                    <Loader2 className="w-10 h-10 animate-spin text-primary" />
                </div>
            )}

            {error && (
                <div className="max-w-xl mx-auto text-center">
                    <p className="text-destructive font-medium bg-destructive/10 p-4 rounded-lg">{error}</p>
                </div>
            )}

            {studentData && (
                <div className="space-y-8">

                    {/* Student Profile Card */}
                    <Card className="border-l-4 border-l-primary shadow-sm bg-primary/5 border-t-0 border-r-0 border-b-0 rounded-r-xl">
                        <CardContent className="p-6 flex items-center justify-between">
                            <div className="flex items-center gap-6">
                                <div className="h-16 w-16 bg-white rounded-full flex items-center justify-center shadow-sm border border-slate-100">
                                    <User className="h-8 w-8 text-primary/60" />
                                </div>
                                <div>
                                    <h2 className="text-2xl font-serif font-bold text-slate-800">{studentData.profile.student_name}</h2>
                                    <div className="flex gap-4 mt-1 text-sm text-slate-600 font-medium">
                                        <span className="flex items-center gap-1.5"><GraduationCap className="h-4 w-4" /> {studentData.profile.department}</span>
                                        <span className="text-slate-300">|</span>
                                        <span>Class: {studentData.profile.class_name}</span>
                                        <span className="text-slate-300">|</span>
                                        <span>ID: {studentData.profile.student_id}</span>
                                    </div>
                                </div>
                            </div>
                            <Button variant="outline" className="gap-2 bg-white hidden md:flex">
                                <Download className="w-4 h-4" /> Download Report
                            </Button>
                        </CardContent>
                    </Card>

                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">

                        {/* Overall Progression Line Chart */}
                        <div className="space-y-4">
                            <h3 className="text-xl font-serif font-bold text-slate-800">Overall Academic Progression</h3>
                            <Card className="border-none shadow-xl shadow-slate-200/50">
                                <CardContent className="h-[350px] pt-6 pr-8">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <LineChart data={progressionData}>
                                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                                            <XAxis dataKey="semester" tick={{ fill: '#64748b', fontSize: 13 }} tickMargin={10} axisLine={false} tickLine={false} />
                                            <YAxis yAxisId="left" domain={[0, 100]} tick={{ fill: '#64748b' }} axisLine={false} tickLine={false} />
                                            <YAxis yAxisId="right" orientation="right" domain={[0, 10]} hide />
                                            <Tooltip
                                                contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                                            />
                                            <Legend wrapperStyle={{ paddingTop: '20px' }} />
                                            <Line yAxisId="left" type="monotone" dataKey="percentage" name="Percentage (%)" stroke="#3b82f6" strokeWidth={3} activeDot={{ r: 8 }} />
                                            <Line yAxisId="right" type="monotone" dataKey="sgpa" name="SGPA" stroke="#8b5cf6" strokeWidth={3} />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </CardContent>
                            </Card>
                        </div>

                        {/* Subject performance for semester */}
                        <div className="space-y-4">
                            <div className="flex justify-between items-end">
                                <h3 className="text-xl font-serif font-bold text-slate-800">Subject Performance</h3>
                                <div className="flex gap-2 bg-slate-100 p-1 rounded-lg">
                                    {studentData.semesters.map(s => (
                                        <button
                                            key={s.semester}
                                            onClick={() => setSelectedSemester(s.semester)}
                                            className={`px-3 py-1 text-xs font-bold rounded-md transition-colors ${selectedSemester === s.semester ? 'bg-white shadow text-primary' : 'text-slate-500 hover:text-slate-800'}`}
                                        >
                                            Sem {s.semester}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            <Card className="border-none shadow-xl shadow-slate-200/50">
                                <CardContent className="h-[350px] pt-6">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <BarChart data={subjectData} layout="vertical" margin={{ left: 20, right: 20 }}>
                                            <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="#e2e8f0" />
                                            <XAxis type="number" domain={[0, 100]} hide />
                                            <YAxis dataKey="subject" type="category" width={80} tick={{ fill: '#475569', fontSize: 12, fontWeight: 500 }} axisLine={false} tickLine={false} />
                                            <Tooltip
                                                cursor={{ fill: 'transparent' }}
                                                contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                                                formatter={(value) => [`${parseFloat(value).toFixed(1)}%`, "Score"]}
                                            />
                                            <Bar dataKey="percentage" name="Score %" fill="#0f172a" radius={[0, 4, 4, 0]} barSize={32}>
                                                <LabelList dataKey="percentage" position="right" formatter={(v) => `${parseFloat(v).toFixed(0)}%`} className="font-bold text-xs" fill="#64748b" />
                                            </Bar>
                                        </BarChart>
                                    </ResponsiveContainer>
                                </CardContent>
                            </Card>

                        </div>
                    </div>

                </div>
            )}
        </div>
    );
}
