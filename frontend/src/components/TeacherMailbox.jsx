import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { useAuth } from '../context/AuthContext';
import { Loader2, Send, Mail, AlertCircle, CheckCircle2 } from 'lucide-react';

export function TeacherMailbox() {
    const { user } = useAuth();

    // Form state
    const [studentId, setStudentId] = useState('');
    const [subject, setSubject] = useState('Academic Defaulter Alert');
    const [body, setBody] = useState('Dear Parent/Guardian,\n\nWe are writing to inform you that your ward has fallen behind in their academic submissions or attendance. Please review their dashboard for more details.\n\nRegards,\nThe Administration');

    // Request State
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');

    const handleSend = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');
        setSuccess('');

        try {
            const token = localStorage.getItem('access_token');
            const response = await fetch('http://127.0.0.1:8000/messages/send', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({
                    student_id: studentId,
                    subject: subject,
                    body: body
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || "Failed to send the alert.");
            }

            setSuccess(`Genuine email successfully dispatched to the parent of ${studentId}!`);
            setStudentId('');
            // Keep default subject/body for rapid sending

        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    if (!user || (user.role !== 'teacher' && user.role !== 'admin')) {
        return (
            <div className="flex flex-col items-center justify-center p-20 text-center">
                <AlertCircle className="w-16 h-16 text-red-500 mb-4" />
                <h2 className="text-2xl font-serif font-bold text-slate-800">Access Denied</h2>
                <p className="text-slate-500 mt-2">Only teachers and administrators can access the dispatch mailbox.</p>
            </div>
        );
    }

    return (
        <div className="max-w-4xl mx-auto p-4 animate-in fade-in duration-500 pb-20 mt-10">
            <div className="mb-8 text-center sm:text-left">
                <h1 className="text-4xl font-serif font-medium text-primary tracking-tight flex items-center justify-center sm:justify-start gap-4">
                    <Mail className="w-8 h-8" /> Defaulter Dispatch
                </h1>
                <p className="text-muted-foreground text-lg font-light mt-2">
                    Send verified, genuine email alerts directly to linked parents.
                </p>
            </div>

            <Card className="border-t-4 border-t-amber-500 shadow-xl shadow-slate-200/50">
                <CardHeader className="bg-amber-50/50 pb-8 rounded-t-xl border-b border-amber-100">
                    <CardTitle className="text-amber-900 flex items-center gap-2">
                        Compose Alert
                    </CardTitle>
                    <CardDescription className="text-amber-700/80">
                        The system will automatically look up the registered parent's email address linked to the Student ID.
                    </CardDescription>
                </CardHeader>
                <CardContent className="p-6 md:p-8 space-y-6">

                    {error && (
                        <div className="p-4 bg-red-50 text-red-700 border border-red-200 rounded-lg flex items-start gap-3 shadow-sm">
                            <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
                            <div>
                                <p className="font-semibold text-sm">Failed to Dispatch</p>
                                <p className="text-sm">{error}</p>
                            </div>
                        </div>
                    )}

                    {success && (
                        <div className="p-4 bg-emerald-50 text-emerald-800 border border-emerald-200 rounded-lg flex items-center gap-3 shadow-sm">
                            <CheckCircle2 className="w-6 h-6 flex-shrink-0 text-emerald-600" />
                            <p className="font-medium">{success}</p>
                        </div>
                    )}

                    <form onSubmit={handleSend} className="space-y-6">

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div className="space-y-2">
                                <Label htmlFor="studentId" className="font-semibold">Target Student ID</Label>
                                <Input
                                    id="studentId"
                                    value={studentId}
                                    onChange={(e) => setStudentId(e.target.value.toUpperCase())}
                                    placeholder="e.g. S001"
                                    className="bg-slate-50 focus-visible:bg-white text-lg font-mono tracking-wider uppercase h-12 border-slate-300"
                                    required
                                />
                                <p className="text-xs text-slate-500">The parent must be approved and linked to this ID to receive the email.</p>
                            </div>

                            <div className="space-y-2">
                                <Label htmlFor="subject" className="font-semibold">Email Subject</Label>
                                <Input
                                    id="subject"
                                    value={subject}
                                    onChange={(e) => setSubject(e.target.value)}
                                    className="bg-slate-50 focus-visible:bg-white h-12 border-slate-300"
                                    required
                                />
                            </div>
                        </div>

                        <div className="space-y-2">
                            <Label htmlFor="body" className="font-semibold">Message Body</Label>
                            <Textarea
                                id="body"
                                value={body}
                                onChange={(e) => setBody(e.target.value)}
                                rows={8}
                                className="bg-slate-50 focus-visible:bg-white resize-none border-slate-300 leading-relaxed"
                                required
                            />
                        </div>

                        <div className="pt-4 flex justify-end">
                            <Button type="submit" className="px-8 py-6 bg-amber-600 hover:bg-amber-700 text-white font-bold text-md shadow-lg shadow-amber-600/20 font-serif" disabled={loading}>
                                {loading ? <Loader2 className="w-5 h-5 animate-spin mr-2" /> : <Send className="w-5 h-5 mr-2" />}
                                {loading ? 'Dispatching...' : 'Dispatch Genuine Alert'}
                            </Button>
                        </div>
                    </form>

                </CardContent>
            </Card>

        </div>
    );
}
