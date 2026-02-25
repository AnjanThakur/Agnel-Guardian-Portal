import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { AlertCircle, Loader2, CheckCircle2, UserCog, UserCheck, Search, Link as LinkIcon, Trash2, X } from 'lucide-react';

export function AdminUsers() {
    const { user, token } = useAuth();
    const [users, setUsers] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const [searchQuery, setSearchQuery] = useState('');

    const [smtpEmail, setSmtpEmail] = useState('');
    const [smtpPassword, setSmtpPassword] = useState('');
    const [savingSmtp, setSavingSmtp] = useState(false);
    const [smtpMessage, setSmtpMessage] = useState('');

    const fetchUsers = async () => {
        setLoading(true);
        setError('');
        try {
            const response = await fetch('http://127.0.0.1:8000/auth/admin/users', {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (!response.ok) throw new Error("Failed to fetch users.");
            const data = await response.json();
            setUsers(data);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const fetchSmtpSettings = async () => {
        try {
            const response = await fetch('http://127.0.0.1:8000/auth/admin/settings/smtp', {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (response.ok) {
                const data = await response.json();
                setSmtpEmail(data.sender_email || '');
                setSmtpPassword(data.app_password || '');
            }
        } catch (err) {
            console.error("Failed to load SMTP settings", err);
        }
    };

    useEffect(() => {
        if (user?.role === 'admin') {
            fetchUsers();
            fetchSmtpSettings();
        }
    }, [user, token]);

    const handleApprove = async (email) => {
        try {
            const response = await fetch(`http://127.0.0.1:8000/auth/admin/approve/${email}`, {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (!response.ok) throw new Error("Failed to approve user.");
            // Refresh list
            fetchUsers();
        } catch (err) {
            alert(err.message);
        }
    };

    const handleUpdateStudents = async (email, newLinkedStudentsArray) => {
        try {
            const response = await fetch(`http://127.0.0.1:8000/auth/admin/users/${email}/students`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({ linked_students: newLinkedStudentsArray })
            });

            if (!response.ok) throw new Error("Failed to update linked students.");
            // Refresh list
            fetchUsers();
        } catch (err) {
            alert(err.message);
        }
    };

    const handleSaveSmtp = async (e) => {
        e.preventDefault();
        setSavingSmtp(true);
        setSmtpMessage('');
        try {
            const response = await fetch('http://127.0.0.1:8000/auth/admin/settings/smtp', {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({ sender_email: smtpEmail, app_password: smtpPassword })
            });
            if (!response.ok) throw new Error("Failed to save SMTP settings");
            setSmtpMessage('SMTP configuration explicitly updated.');
            // Clear message after 3 seconds
            setTimeout(() => setSmtpMessage(''), 3000);
        } catch (err) {
            setSmtpMessage(err.message);
        } finally {
            setSavingSmtp(false);
        }
    };

    if (user?.role !== 'admin') {
        return (
            <div className="flex flex-col items-center justify-center p-20 text-center">
                <AlertCircle className="w-16 h-16 text-red-500 mb-4" />
                <h2 className="text-2xl font-serif font-bold text-slate-800">Access Denied</h2>
                <p className="text-slate-500 mt-2">Only administrators can access the User Management dashboard.</p>
            </div>
        );
    }

    const filteredUsers = users.filter(u =>
        u.email.toLowerCase().includes(searchQuery.toLowerCase()) ||
        u.role.toLowerCase().includes(searchQuery.toLowerCase())
    );

    return (
        <div className="max-w-6xl mx-auto p-4 animate-in fade-in duration-500 pb-20 mt-10">
            <div className="mb-8 text-center sm:text-left flex flex-col md:flex-row md:items-end justify-between gap-6">
                <div>
                    <h1 className="text-4xl font-serif font-medium text-primary tracking-tight flex items-center justify-center sm:justify-start gap-4">
                        <UserCog className="w-8 h-8" /> User Management
                    </h1>
                    <p className="text-muted-foreground text-lg font-light mt-2">
                        Approve new signups and manage assigned Student accounts across the portal.
                    </p>
                </div>
                <Button onClick={fetchUsers} disabled={loading} variant="outline" className="shadow-sm bg-white">
                    {loading ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : "Refresh Table"}
                </Button>
            </div>

            {error && (
                <div className="mb-6 p-4 bg-red-50 text-red-700 border border-red-200 rounded-lg flex items-start gap-3 shadow-sm">
                    <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
                    <p className="font-medium text-sm">{error}</p>
                </div>
            )}

            <Card className="mb-8 border-t-4 border-t-amber-500 shadow-md">
                <CardHeader className="bg-amber-50/50 pb-4 border-b border-amber-100">
                    <CardTitle className="text-amber-900 text-lg flex items-center gap-2">
                        Global Mailing Configuration
                    </CardTitle>
                    <CardDescription className="text-amber-700/80">
                        Set the authenticated System Sender Email and App Password here so educators don't have to input credentials during dispatch.
                    </CardDescription>
                </CardHeader>
                <CardContent className="p-6">
                    <form onSubmit={handleSaveSmtp} className="flex flex-col md:flex-row gap-6 items-end">
                        <div className="w-full space-y-2">
                            <Label htmlFor="smtpEmail" className="font-semibold text-slate-700">Sender Auth Email</Label>
                            <Input
                                id="smtpEmail"
                                type="email"
                                value={smtpEmail}
                                onChange={(e) => setSmtpEmail(e.target.value)}
                                className="bg-slate-50 border-slate-300"
                                placeholder="admin@agnel.edu"
                                required
                            />
                        </div>
                        <div className="w-full space-y-2">
                            <Label htmlFor="smtpPassword" className="font-semibold text-slate-700">App Password</Label>
                            <Input
                                id="smtpPassword"
                                type="password"
                                value={smtpPassword}
                                onChange={(e) => setSmtpPassword(e.target.value)}
                                className="bg-slate-50 border-slate-300"
                                placeholder="16-character SMTP password"
                                required
                            />
                        </div>
                        <Button type="submit" className="h-10 hover:bg-amber-600 bg-amber-500 text-white min-w-[140px] shadow-sm font-semibold" disabled={savingSmtp}>
                            {savingSmtp ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : <CheckCircle2 className="w-4 h-4 mr-2" />}
                            {savingSmtp ? "Saving..." : "Save Config"}
                        </Button>
                    </form>
                    {smtpMessage && <p className={`mt-4 text-sm font-semibold flex items-center gap-2 ${smtpMessage.includes('Failed') ? 'text-red-600' : 'text-emerald-600'}`}>
                        {smtpMessage.includes('Failed') ? <AlertCircle className="w-4 h-4" /> : <CheckCircle2 className="w-4 h-4" />}
                        {smtpMessage}
                    </p>}
                </CardContent>
            </Card>

            <Card className="border-t-4 border-t-primary shadow-xl shadow-slate-200/50 overflow-hidden">
                <div className="p-4 border-b border-slate-100 bg-slate-50 flex items-center gap-3">
                    <Search className="w-5 h-5 text-slate-400" />
                    <Input
                        placeholder="Search by email or role..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="max-w-sm bg-white border-slate-200"
                    />
                </div>
                <div className="overflow-x-auto">
                    <table className="w-full text-sm text-left">
                        <thead className="bg-slate-50 text-slate-600 font-semibold border-b border-slate-200">
                            <tr>
                                <th className="px-6 py-4">Account Email</th>
                                <th className="px-6 py-4">Role</th>
                                <th className="px-6 py-4">Status</th>
                                <th className="px-6 py-4">Linked Students</th>
                                <th className="px-6 py-4 text-right">Actions</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100">
                            {filteredUsers.map((u) => (
                                <tr key={u.email} className="hover:bg-slate-50/50 transition-colors">
                                    <td className="px-6 py-4 font-medium text-slate-800">{u.email}</td>
                                    <td className="px-6 py-4">
                                        <Badge variant={u.role === 'admin' ? 'default' : u.role === 'teacher' ? 'secondary' : 'outline'} className="capitalize">
                                            {u.role}
                                        </Badge>
                                    </td>
                                    <td className="px-6 py-4">
                                        {u.is_approved ? (
                                            <span className="flex items-center gap-1.5 text-emerald-600 font-semibold text-xs">
                                                <CheckCircle2 className="w-4 h-4" /> Approved
                                            </span>
                                        ) : (
                                            <span className="flex items-center gap-1.5 text-amber-600 font-semibold text-xs">
                                                <AlertCircle className="w-4 h-4" /> Pending
                                            </span>
                                        )}
                                    </td>
                                    <td className="px-6 py-4">
                                        {(u.role === 'parent' || u.role === 'student') ? (
                                            <div className="flex flex-wrap gap-1">
                                                {u.linked_students?.length > 0 ? (
                                                    u.linked_students.map(id => (
                                                        <span key={id} className="bg-indigo-50 text-indigo-700 px-2 py-0.5 rounded text-xs font-mono font-bold border border-indigo-100">
                                                            {id}
                                                        </span>
                                                    ))
                                                ) : (
                                                    <span className="text-slate-400 italic text-xs">None linked</span>
                                                )}
                                            </div>
                                        ) : (
                                            <span className="text-slate-300">-</span>
                                        )}
                                    </td>
                                    <td className="px-6 py-4 text-right space-x-2">
                                        {!u.is_approved && (
                                            <Button
                                                size="sm"
                                                onClick={() => handleApprove(u.email)}
                                                className="bg-emerald-600 hover:bg-emerald-700 h-8 text-xs font-bold"
                                            >
                                                <UserCheck className="w-3 h-3 mr-1.5" /> Approve
                                            </Button>
                                        )}

                                        {(u.role === 'parent' || u.role === 'student') && (
                                            <StudentManagerModal
                                                userEmail={u.email}
                                                currentStudents={u.linked_students || []}
                                                onSave={handleUpdateStudents}
                                            />
                                        )}
                                    </td>
                                </tr>
                            ))}
                            {filteredUsers.length === 0 && (
                                <tr>
                                    <td colSpan="5" className="px-6 py-12 text-center text-slate-500">
                                        No users found matching your search.
                                    </td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>
            </Card>
        </div>
    );
}

function StudentManagerModal({ userEmail, currentStudents, onSave }) {
    const [isOpen, setIsOpen] = useState(false);
    const [tempList, setTempList] = useState([...currentStudents]);
    const [newId, setNewId] = useState('');

    // Reset when opened
    useEffect(() => {
        if (isOpen) {
            setTempList([...currentStudents]);
            setNewId('');
        }
    }, [isOpen, currentStudents]);

    const handleAdd = () => {
        const val = newId.trim().toUpperCase();
        if (val && !tempList.includes(val)) {
            setTempList([...tempList, val]);
        }
        setNewId('');
    };

    const handleRemove = (idToRemove) => {
        setTempList(tempList.filter(id => id !== idToRemove));
    };

    const handleSave = () => {
        onSave(userEmail, tempList);
        setIsOpen(false);
    };

    return (
        <>
            <Button size="sm" variant="outline" className="h-8 text-xs text-indigo-700 border-indigo-200 hover:bg-indigo-50" onClick={() => setIsOpen(true)}>
                <LinkIcon className="w-3 h-3 mr-1.5" /> Manage Links
            </Button>

            {isOpen && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-transparent backdrop-blur-sm p-4">
                    <div className="absolute inset-0 bg-slate-900/40" onClick={() => setIsOpen(false)}></div>
                    <div className="relative bg-white rounded-xl shadow-2xl w-full max-w-md overflow-hidden animate-in fade-in zoom-in-95 duration-200 border border-slate-200">
                        <div className="px-6 py-4 border-b border-slate-100 flex justify-between items-start">
                            <div>
                                <h3 className="text-lg font-semibold text-slate-900">Manage Links for <span className="text-primary">{userEmail}</span></h3>
                                <p className="text-sm text-slate-500 mt-1">Manually assign or revoke access to student records for this account.</p>
                            </div>
                            <Button variant="ghost" size="icon" onClick={() => setIsOpen(false)} className="-mt-1 -mr-2 text-slate-400 hover:text-slate-600">
                                <X className="w-4 h-4" />
                            </Button>
                        </div>

                        <div className="p-6 space-y-6">
                            <div className="space-y-4">
                                <Label className="text-slate-700 font-semibold">Currently Linked Students</Label>
                                {tempList.length > 0 ? (
                                    <div className="flex flex-col gap-2 bg-slate-50 p-3 rounded-lg border border-slate-100 max-h-[200px] overflow-y-auto">
                                        {tempList.map(id => (
                                            <div key={id} className="flex items-center justify-between bg-white px-3 py-2 rounded-md border border-slate-200 shadow-sm">
                                                <span className="font-mono font-bold text-slate-700">{id}</span>
                                                <Button
                                                    size="sm"
                                                    variant="ghost"
                                                    className="h-7 w-7 p-0 text-red-500 hover:text-red-700 hover:bg-red-50"
                                                    onClick={() => handleRemove(id)}
                                                >
                                                    <Trash2 className="w-4 h-4" />
                                                </Button>
                                            </div>
                                        ))}
                                    </div>
                                ) : (
                                    <div className="p-4 bg-slate-50 rounded-lg border border-dashed border-slate-300 text-center text-sm text-slate-500">
                                        No students are currently linked.
                                    </div>
                                )}
                            </div>

                            <div className="flex gap-2 items-end">
                                <div className="flex-1 space-y-2">
                                    <Label htmlFor="add-student" className="text-xs font-semibold text-slate-600">Add Student ID</Label>
                                    <Input
                                        id="add-student"
                                        value={newId}
                                        onChange={(e) => setNewId(e.target.value.toUpperCase())}
                                        placeholder="E.g. S003"
                                        className="font-mono uppercase h-10 border-slate-300"
                                        onKeyDown={(e) => e.key === 'Enter' && handleAdd()}
                                    />
                                </div>
                                <Button type="button" onClick={handleAdd} className="h-10 px-4 bg-slate-800 hover:bg-slate-900 text-white shadow-sm">Add</Button>
                            </div>
                        </div>

                        <div className="px-6 py-4 bg-slate-50 border-t border-slate-100 flex justify-end gap-3">
                            <Button variant="outline" onClick={() => setIsOpen(false)} className="bg-white border-slate-200 hover:bg-slate-100">Cancel</Button>
                            <Button onClick={handleSave} className="bg-primary hover:bg-primary/90 text-white shadow-md">Save Changes</Button>
                        </div>
                    </div>
                </div>
            )}
        </>
    );
}
