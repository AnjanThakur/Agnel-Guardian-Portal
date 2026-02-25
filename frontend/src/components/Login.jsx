import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Loader2, ShieldCheck, UserPlus } from 'lucide-react';

export function Login() {
    const { login } = useAuth();
    const navigate = useNavigate();

    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    // Login State
    const [loginEmail, setLoginEmail] = useState('');
    const [loginPassword, setLoginPassword] = useState('');

    // Register State
    const [regEmail, setRegEmail] = useState('');
    const [regPassword, setRegPassword] = useState('');
    const [regRole, setRegRole] = useState('parent');
    const [successMsg, setSuccessMsg] = useState('');

    const handleLogin = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');

        try {
            const formData = new URLSearchParams();
            formData.append('username', loginEmail);
            formData.append('password', loginPassword);

            const response = await fetch('http://127.0.0.1:8000/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                body: formData
            });

            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.detail || 'Login failed');
            }

            const data = await response.json();
            login(data.access_token);

            // Recheck session happens implicitly in context, but we fetch /me to know where to route
            const meRes = await fetch('http://127.0.0.1:8000/auth/me', {
                headers: { 'Authorization': `Bearer ${data.access_token}` }
            });
            const meData = await meRes.json();

            if (meData.role === 'admin' || meData.role === 'teacher') {
                navigate('/analytics');
            } else {
                navigate('/parent-portal');
            }

        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleRegister = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');
        setSuccessMsg('');

        try {
            const response = await fetch('http://127.0.0.1:8000/auth/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    email: regEmail,
                    password: regPassword,
                    role: regRole,
                    linked_students: []
                })
            });

            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.detail || 'Registration failed');
            }

            setSuccessMsg("Registration successful! Your account is pending Admin approval. You will not be able to log in until approved.");
            setRegEmail('');
            setRegPassword('');

        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-[80vh] flex items-center justify-center bg-slate-50 p-4">
            <Card className="w-full max-w-md shadow-2xl shadow-slate-200/50 border-none">
                <CardHeader className="space-y-1 text-center pb-8 border-b border-slate-100 bg-white rounded-t-xl">
                    <div className="mx-auto bg-primary/10 p-4 rounded-full w-20 h-20 flex items-center justify-center mb-4">
                        <ShieldCheck className="w-10 h-10 text-primary" />
                    </div>
                    <CardTitle className="text-3xl font-serif font-bold text-slate-800 tracking-tight">Portal Access</CardTitle>
                    <CardDescription className="text-slate-500 font-medium text-md">
                        Securely log into the Agnel Guardian Portal
                    </CardDescription>
                </CardHeader>

                <CardContent className="p-6 bg-white rounded-b-xl">
                    <Tabs defaultValue="login" className="w-full">
                        <TabsList className="grid w-full grid-cols-2 mb-8 bg-slate-100/50 p-1">
                            <TabsTrigger value="login" className="data-[state=active]:bg-white data-[state=active]:shadow-sm rounded-md transition-all font-semibold">Sign In</TabsTrigger>
                            <TabsTrigger value="register" className="data-[state=active]:bg-white data-[state=active]:shadow-sm rounded-md transition-all font-semibold">Sign Up</TabsTrigger>
                        </TabsList>

                        <TabsContent value="login" className="animate-in fade-in zoom-in-95 duration-200">
                            {error && <div className="mb-6 p-3 bg-red-50 text-red-600 border border-red-100 rounded-lg text-sm font-medium flex items-center gap-2 px-4 shadow-sm">{error}</div>}
                            <form onSubmit={handleLogin} className="space-y-5">
                                <div className="space-y-2">
                                    <Label htmlFor="email">Email</Label>
                                    <Input
                                        id="email"
                                        type="email"
                                        value={loginEmail}
                                        onChange={(e) => setLoginEmail(e.target.value)}
                                        className="bg-slate-50/50 border-slate-200 focus-visible:ring-primary focus-visible:bg-white transition-colors"
                                        placeholder="Enter your email"
                                        required
                                    />
                                </div>
                                <div className="space-y-2">
                                    <div className="flex items-center justify-between">
                                        <Label htmlFor="password">Password</Label>
                                        <span className="text-xs text-primary font-medium cursor-pointer hover:underline">Forgot password?</span>
                                    </div>
                                    <Input
                                        id="password"
                                        type="password"
                                        value={loginPassword}
                                        onChange={(e) => setLoginPassword(e.target.value)}
                                        className="bg-slate-50/50 border-slate-200 focus-visible:ring-primary focus-visible:bg-white transition-colors"
                                        placeholder="••••••••"
                                        required
                                    />
                                </div>
                                <Button type="submit" className="w-full py-6 text-md font-semibold mt-2 shadow-lg shadow-primary/20 hover:shadow-primary/30 transition-all font-serif" disabled={loading}>
                                    {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : 'Log Into Portal'}
                                </Button>
                            </form>
                        </TabsContent>

                        <TabsContent value="register" className="animate-in fade-in zoom-in-95 duration-200">
                            {error && <div className="mb-4 p-3 bg-red-50 text-red-600 border border-red-100 rounded-lg text-sm font-medium">{error}</div>}
                            {successMsg && <div className="mb-4 p-4 bg-emerald-50 text-emerald-700 border border-emerald-100 rounded-lg text-sm font-semibold">{successMsg}</div>}

                            <form onSubmit={handleRegister} className="space-y-5">
                                <div className="space-y-2">
                                    <Label htmlFor="regEmail">Email Address</Label>
                                    <Input
                                        id="regEmail"
                                        type="email"
                                        value={regEmail}
                                        onChange={(e) => setRegEmail(e.target.value)}
                                        placeholder="name@example.com"
                                        required
                                    />
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="regPassword">Strong Password</Label>
                                    <Input
                                        id="regPassword"
                                        type="password"
                                        value={regPassword}
                                        onChange={(e) => setRegPassword(e.target.value)}
                                        placeholder="Create a secure password"
                                        required
                                    />
                                </div>
                                <div className="space-y-3">
                                    <Label>I am registering as a:</Label>
                                    <div className="flex gap-4 p-1">
                                        <label className="flex items-center gap-2 cursor-pointer border rounded-lg px-4 py-3 flex-1 hover:bg-slate-50 transition-colors">
                                            <input
                                                type="radio"
                                                name="role"
                                                value="parent"
                                                checked={regRole === 'parent'}
                                                onChange={(e) => setRegRole(e.target.value)}
                                                className="accent-primary w-4 h-4"
                                            />
                                            <span className="font-medium text-sm">Parent</span>
                                        </label>
                                        <label className="flex items-center gap-2 cursor-pointer border rounded-lg px-4 py-3 flex-1 hover:bg-slate-50 transition-colors">
                                            <input
                                                type="radio"
                                                name="role"
                                                value="student"
                                                checked={regRole === 'student'}
                                                onChange={(e) => setRegRole(e.target.value)}
                                                className="accent-primary w-4 h-4"
                                            />
                                            <span className="font-medium text-sm">Student</span>
                                        </label>
                                    </div>
                                    <p className="text-xs text-slate-500 mt-2">
                                        Accounts require manual administrator approval before access is granted.
                                    </p>
                                </div>
                                <Button type="submit" variant="outline" className="w-full py-6 text-md font-semibold mt-4 border-2 hover:bg-slate-50 font-serif" disabled={loading}>
                                    {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <span className="flex items-center gap-2"><UserPlus className="w-4 h-4" /> Submit Registration</span>}
                                </Button>
                            </form>
                        </TabsContent>
                    </Tabs>
                </CardContent>
            </Card>
        </div>
    );
}
