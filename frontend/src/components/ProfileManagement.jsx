import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { useAuth } from '../context/AuthContext';
import { Loader2, UserCog, Link as LinkIcon, Save, CheckCircle2, AlertCircle } from 'lucide-react';

export function ProfileManagement() {
    const { user, checkSession } = useAuth();

    // Form state
    const [email, setEmail] = useState(user?.email || '');
    const [password, setPassword] = useState('');
    const [newStudentId, setNewStudentId] = useState('');

    // Request State
    const [loadingSettings, setLoadingSettings] = useState(false);
    const [loadingLink, setLoadingLink] = useState(false);
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');

    if (!user) return null;

    const handleUpdateProfile = async (e) => {
        e.preventDefault();
        setLoadingSettings(true);
        setError('');
        setSuccess('');

        const updates = {};
        if (email !== user.email) updates.email = email;
        if (password) updates.password = password;

        if (Object.keys(updates).length === 0) {
            setLoadingSettings(false);
            return;
        }

        try {
            const token = localStorage.getItem('access_token');
            const response = await fetch('http://127.0.0.1:8000/auth/profile', {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify(updates)
            });

            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.detail || "Failed to update profile.");
            }

            setSuccess("Security settings updated successfully.");
            setPassword(''); // Clear password field
            await checkSession(); // Refresh global user state

        } catch (err) {
            setError(err.message);
        } finally {
            setLoadingSettings(false);
        }
    };

    const handleLinkStudent = async (e) => {
        e.preventDefault();
        if (!newStudentId) return;

        setLoadingLink(true);
        setError('');
        setSuccess('');

        try {
            const token = localStorage.getItem('access_token');
            const response = await fetch('http://127.0.0.1:8000/auth/link_student', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({ student_id: newStudentId.toUpperCase() })
            });

            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.detail || "Failed to link student.");
            }

            setSuccess(`Successfully linked Student ID: ${newStudentId.toUpperCase()}!`);
            setNewStudentId('');
            await checkSession(); // Refresh global user state to show the new badge

        } catch (err) {
            setError(err.message);
        } finally {
            setLoadingLink(false);
        }
    };

    return (
        <div className="max-w-4xl mx-auto p-4 animate-in fade-in duration-500 pb-20 mt-10">
            <div className="mb-8 text-center sm:text-left">
                <h1 className="text-4xl font-serif font-medium text-primary tracking-tight flex items-center justify-center sm:justify-start gap-4">
                    <UserCog className="w-8 h-8" /> Profile Management
                </h1>
                <p className="text-muted-foreground text-lg font-light mt-2">
                    Update your security settings and manage linked student accounts.
                </p>
            </div>

            {error && (
                <div className="mb-6 p-4 bg-red-50 text-red-700 border border-red-200 rounded-lg flex items-start gap-3 shadow-sm">
                    <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
                    <p className="font-medium text-sm">{error}</p>
                </div>
            )}

            {success && (
                <div className="mb-6 p-4 bg-emerald-50 text-emerald-800 border border-emerald-200 rounded-lg flex items-center gap-3 shadow-sm">
                    <CheckCircle2 className="w-6 h-6 flex-shrink-0 text-emerald-600" />
                    <p className="font-medium">{success}</p>
                </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">

                {/* Security Settings */}
                <Card className="border-t-4 border-t-primary shadow-xl shadow-slate-200/50">
                    <CardHeader className="bg-slate-50/50 pb-6 border-b border-slate-100">
                        <CardTitle className="text-slate-800 flex items-center gap-2 text-xl">
                            Account Security
                        </CardTitle>
                        <CardDescription>Update your email address to receive notifications.</CardDescription>
                    </CardHeader>
                    <CardContent className="p-6">
                        <form onSubmit={handleUpdateProfile} className="space-y-5">
                            <div className="space-y-2">
                                <Label htmlFor="email" className="font-semibold text-slate-700">Email Address</Label>
                                <Input
                                    id="email"
                                    type="email"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    className="bg-slate-50 h-11"
                                    required
                                />
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="password" className="font-semibold text-slate-700">New Password (Optional)</Label>
                                <Input
                                    id="password"
                                    type="password"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    placeholder="Leave blank to keep current"
                                    className="bg-slate-50 h-11"
                                />
                            </div>
                            <Button type="submit" className="w-full h-11 font-semibold mt-4 shadow-md font-serif" disabled={loadingSettings}>
                                {loadingSettings ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : <Save className="w-4 h-4 mr-2" />}
                                Save Changes
                            </Button>
                        </form>
                    </CardContent>
                </Card>

                {/* Linked Students (Only Parent/Student) */}
                {user.role === 'parent' || user.role === 'student' ? (
                    <Card className="border-t-4 border-t-indigo-500 shadow-xl shadow-slate-200/50">
                        <CardHeader className="bg-indigo-50/50 pb-6 border-b border-indigo-100">
                            <CardTitle className="text-indigo-900 flex items-center gap-2 text-xl">
                                Linked Wards
                            </CardTitle>
                            <CardDescription className="text-indigo-700/80">Manage which students appear on your Parent Dashboard.</CardDescription>
                        </CardHeader>
                        <CardContent className="p-6">

                            <div className="mb-6">
                                <Label className="text-slate-500 mb-3 block text-xs uppercase tracking-wider font-bold">Currently Linked</Label>
                                {user.linked_students?.length > 0 ? (
                                    <div className="flex flex-wrap gap-2">
                                        {user.linked_students.map(id => (
                                            <span key={id} className="bg-white border-2 border-indigo-100 text-indigo-700 font-mono px-3 py-1.5 rounded-lg text-sm font-bold shadow-sm inline-flex items-center gap-2">
                                                <UserCog className="w-3 h-3 text-indigo-400" /> {id}
                                            </span>
                                        ))}
                                    </div>
                                ) : (
                                    <p className="text-sm text-slate-400 italic">No students linked yet.</p>
                                )}
                            </div>

                            {user.role === 'parent' && (
                                <form onSubmit={handleLinkStudent} className="space-y-4 pt-6 border-t border-slate-100">
                                    <div className="space-y-2">
                                        <Label htmlFor="newStudentId" className="font-semibold text-slate-700">Add New Student ID</Label>
                                        <div className="flex gap-3">
                                            <Input
                                                id="newStudentId"
                                                value={newStudentId}
                                                onChange={(e) => setNewStudentId(e.target.value.toUpperCase())}
                                                className="bg-slate-50 h-11 font-mono uppercase tracking-wider"
                                                placeholder="e.g. S002"
                                                required
                                            />
                                            <Button type="submit" variant="secondary" className="h-11 px-6 bg-indigo-100 hover:bg-indigo-200 text-indigo-700 font-bold" disabled={loadingLink}>
                                                {loadingLink ? <Loader2 className="w-4 h-4 animate-spin" /> : <LinkIcon className="w-4 h-4" />}
                                            </Button>
                                        </div>
                                    </div>
                                </form>
                            )}

                        </CardContent>
                    </Card>
                ) : null}

            </div>
        </div>
    );
}
