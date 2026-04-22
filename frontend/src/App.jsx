import React, { useState, useEffect } from 'react'
import { Routes, Route, Navigate, Link, useLocation } from 'react-router-dom'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { FileUpload } from '@/components/FileUpload'
import { ResultsDisplay } from '@/components/ResultsDisplay'
import { SidebarList } from '@/components/SidebarList'
import { AISummaryReport } from '@/components/AISummaryReport'
import { AnalyticsView } from '@/components/AnalyticsView'
import { ParentDashboard } from '@/components/ParentDashboard'

// New Auth Components
import { Login } from '@/components/Login'
import { TeacherMailbox } from '@/components/TeacherMailbox'
import { ProfileManagement } from '@/components/ProfileManagement'
import { AdminUsers } from '@/components/AdminUsers'
import { useAuth } from '@/context/AuthContext'
import { useExtraction } from '@/context/ExtractionContext'

import { Loader2, ChevronDown, GraduationCap, ShieldCheck, RefreshCw, Trash2, LineChart, Upload, Plus, Users, UserCog, Mail, LogOut } from 'lucide-react'

// Elegant Logo Component
const Logo = () => (
  <div className="flex items-center gap-2">
    <div className="flex h-10 w-10 items-center justify-center rounded-sm bg-primary text-primary-foreground shadow-sm">
      <GraduationCap className="h-6 w-6" />
    </div>
  </div>
)

// Global Protected Route Wrapper
const ProtectedRoute = ({ children, allowedRoles }) => {
  const { user, loading } = useAuth();
  const location = useLocation();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  if (allowedRoles && !allowedRoles.includes(user.role)) {
    // Redirect based on role if they try to access something forbidden
    if (user.role === 'parent' || user.role === 'student') return <Navigate to="/parent-portal" replace />;
    if (user.role === 'admin' || user.role === 'teacher') return <Navigate to="/analytics" replace />;
    return <Navigate to="/login" replace />;
  }

  return children;
};

// Extracted Extraction View (Legacy feature but protected for Teachers)
function ExtractionView() {
  const fileInputRef = React.useRef(null)
  const { 
    files, setFiles, 
    results, setResults, 
    aiSummary, setAiSummary, 
    selectedIndex, setSelectedIndex, 
    clearAll 
  } = useExtraction()

  const [preview, setPreview] = useState(null)
  const [mode, setMode] = useState('pta_free')
  const [template, setTemplate] = useState('')
  const [debug, setDebug] = useState(false)
  const [department, setDepartment] = useState('CSE')
  const [className, setClassName] = useState('')
  const [batchProcessing, setBatchProcessing] = useState(false)
  const [isGeneratingSummary, setIsGeneratingSummary] = useState(false)

  // --- HANDLER: Append Files ---
  const handleAppendFiles = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const appended = Array.from(e.target.files)
      const startIndex = files.length
      setFiles(prev => [...prev, ...appended])
      setResults(prev => {
        const next = { ...prev }
        appended.forEach((_, i) => { next[startIndex + i] = { status: 'pending' } })
        return next
      })
      e.target.value = null
    }
  }

  // --- LOGIC: Helper to process ONE file ---
  const processSingleFile = async (file, index) => {
    try {
      const toBase64 = (f) => new Promise((resolve, reject) => {
        const r = new FileReader()
        r.readAsDataURL(f)
        r.onload = () => resolve(r.result)
        r.onerror = error => reject(error)
      })
      const base64 = await toBase64(file)

      const url = mode === "pta" ? "/ocr/pta" : mode === "pta_free" ? "/ocr/pta_free" : "/ocr/auto"

      const payload = {
        imageBase64: base64,
        template: template || null,
        debug: debug,
        department: department,
        class_name: className
      }

      setResults(prev => ({ ...prev, [index]: { ...prev[index], status: 'loading' } }))

      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })

      if (!response.ok) throw new Error(`Server error: ${response.status}`)
      const data = await response.json()

      setResults(prev => ({ ...prev, [index]: { status: 'success', data: data } }))
      return data

    } catch (err) {
      console.error(`Error processing file ${index}:`, err)
      setResults(prev => ({ ...prev, [index]: { status: 'error', error: err.message || "Failed" } }))
      return null
    }
  }

  // --- LOGIC: Batch Loop ---
  const runBatchOCR = async () => {
    setBatchProcessing(true)
    setAiSummary(null)

    const commentsByIndex = new Array(files.length).fill(null);
    const queue = files.map((file, index) => ({ file, index }));
    const activeWorkers = [];
    const CONCURRENCY_LIMIT = 5;

    const worker = async () => {
      while (queue.length > 0) {
        const { file, index } = queue.shift();
        if (results[index]?.status === 'success') {
          if (results[index].data?.comments) commentsByIndex[index] = results[index].data.comments;
          continue;
        }
        const data = await processSingleFile(file, index);
        if (data?.comments) commentsByIndex[index] = data.comments;
      }
    };

    for (let i = 0; i < CONCURRENCY_LIMIT; i++) activeWorkers.push(worker());
    await Promise.all(activeWorkers);

    const collectedComments = commentsByIndex.filter(c => c !== null);
    setBatchProcessing(false)

    if (collectedComments.length > 0) {
      setIsGeneratingSummary(true)
      try {
        const response = await fetch('/analysis/summarize_ai', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ comments: collectedComments })
        })
        const data = await response.json()
        setAiSummary(data)
      } catch (e) {
        setAiSummary({ error: 'Failed to generate summary', details: e.message })
      } finally {
        setIsGeneratingSummary(false)
      }
    }
  }

  // clearAll is now handled by Context

  const handleFilesSelected = (newFiles) => {
    setFiles(newFiles)
    const initialResults = {}
    newFiles.forEach((_, idx) => { initialResults[idx] = { status: 'pending' } })
    setResults(initialResults)
    setSelectedIndex(0)
    setAiSummary(null)
  }

  // Auto preview hook
  useEffect(() => {
    if (files.length > 0 && files[selectedIndex]) {
      const file = files[selectedIndex]
      const reader = new FileReader()
      reader.onload = (e) => setPreview(e.target.result)
      reader.readAsDataURL(file)
    } else {
      setPreview(null)
    }
  }, [selectedIndex, files])

  const hasFiles = files.length > 0;
  const currentResult = results[selectedIndex] || {}

  return (
    <div className="space-y-8">
      {!hasFiles ? (
        <div className="max-w-3xl mx-auto space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500 mt-10">
          <div className="text-center space-y-4">
            <h1 className="text-4xl font-serif font-medium text-primary tracking-tight">
              Start Extraction
            </h1>
            <p className="text-muted-foreground text-lg font-light">
              Upload batches of PTA forms in images or PDFs.
            </p>
          </div>
          <Card className="border-none shadow-2xl shadow-slate-200/50">
            <FileUpload onFilesSelected={handleFilesSelected} />
          </Card>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start animate-in fade-in duration-500">
          <div className="lg:col-span-4 space-y-6 flex flex-col h-[calc(100vh-140px)] sticky top-24">
            <div className="flex items-center gap-2">
              <Button variant="outline" size="icon" onClick={clearAll} disabled={batchProcessing} title="Clear All">
                <Trash2 className="w-4 h-4 text-muted-foreground" />
              </Button>

              <div className="w-px h-6 bg-border mx-1" />

              <input
                type="file"
                multiple
                className="hidden"
                ref={fileInputRef}
                onChange={handleAppendFiles}
                accept="image/*,application/pdf"
              />
              <Button
                variant="outline"
                size="icon"
                onClick={() => fileInputRef.current?.click()}
                disabled={batchProcessing}
                title="Add More Files"
              >
                <Plus className="w-4 h-4 text-muted-foreground" />
              </Button>

              <Button
                size="lg"
                onClick={runBatchOCR}
                disabled={batchProcessing}
                className="flex-1 font-serif tracking-wide shadow-lg shadow-primary/20"
              >
                {batchProcessing && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                {batchProcessing ? 'Process New' : 'Run Extraction'}
              </Button>
            </div>
            <div className="flex-1 min-h-0">
              <SidebarList files={files} activeIndex={selectedIndex} onSelect={setSelectedIndex} results={results} />
            </div>
          </div>
          <div className="lg:col-span-8 space-y-6">
            <div className="min-h-[600px]">
              <ResultsDisplay data={currentResult.data || {}} previewUrl={preview} />

              {currentResult.status === 'error' && (
                <div className="mt-4 p-4 border border-destructive/20 bg-destructive/5 rounded-lg text-destructive text-sm font-medium">
                  Analysis Failed: {currentResult.error}
                </div>
              )}

              {currentResult.status === 'pending' && !currentResult.error && (
                <div className="h-64 flex flex-col items-center justify-center text-muted-foreground border-2 border-dashed border-slate-100 rounded-xl bg-slate-50/50 mt-4">
                  <p className="font-serif italic">Ready to process.</p>
                  <p className="text-xs mt-1">Click "Run Extraction" to start batch.</p>
                </div>
              )}

              {currentResult.status === 'loading' && (
                <div className="h-64 flex flex-col items-center justify-center text-primary mt-4">
                  <Loader2 className="w-8 h-8 animate-spin mb-4 text-accent" />
                  <p className="font-serif">Analyzing document...</p>
                </div>
              )}

              {isGeneratingSummary && (
                <div className="mt-6 p-6 border-2 border-dashed border-primary/30 rounded-xl bg-primary/5 text-center">
                  <span className="text-primary font-serif">Generating Summary...</span>
                </div>
              )}
              {aiSummary && !isGeneratingSummary && (
                <div className="mt-6">
                  <AISummaryReport data={aiSummary} />
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function App() {
  const { user, logout, loading } = useAuth();
  const location = useLocation();

  if (loading) return <div className="min-h-screen flex items-center justify-center"><Loader2 className="w-8 h-8 animate-spin text-primary" /></div>;

  // Header Nav Links based on Role
  const NavLinks = () => {
    if (!user) return null;

    const isTeacher = user.role === 'admin' || user.role === 'teacher';

    return (
      <div className="flex flex-wrap items-center gap-2 bg-slate-100 p-1.5 rounded-xl border border-slate-200">
        {isTeacher && (
          <>
            {user.role === 'admin' && (
              <Link to="/admin/users">
                <Button variant={location.pathname === '/admin/users' ? 'default' : 'ghost'} size="sm" className="gap-2 font-medium">
                  <UserCog className="w-4 h-4" /> Manage Users
                </Button>
              </Link>
            )}
            <Link to="/extraction">
              <Button variant={location.pathname === '/extraction' ? 'default' : 'ghost'} size="sm" className="gap-2 font-medium">
                <Upload className="w-4 h-4" /> Extraction
              </Button>
            </Link>
            <Link to="/analytics">
              <Button variant={location.pathname === '/analytics' ? 'default' : 'ghost'} size="sm" className="gap-2 font-medium">
                <LineChart className="w-4 h-4" /> Analytics
              </Button>
            </Link>
            <Link to="/mailbox">
              <Button variant={location.pathname === '/mailbox' ? 'default' : 'ghost'} size="sm" className="gap-2 font-medium">
                <Mail className="w-4 h-4" /> Mailbox
              </Button>
            </Link>
          </>
        )}

        {(!isTeacher) && (
          <Link to="/parent-portal">
            <Button variant={location.pathname === '/parent-portal' ? 'default' : 'ghost'} size="sm" className="gap-2 font-medium">
              <Users className="w-4 h-4" /> Parent Portal
            </Button>
          </Link>
        )}

        <div className="w-px h-6 bg-slate-300 mx-1 hidden sm:block" />

        <Link to="/profile">
          <Button variant={location.pathname === '/profile' ? 'default' : 'ghost'} size="sm" className="gap-2 font-medium">
            <UserCog className="w-4 h-4" /> Profile
          </Button>
        </Link>
        <Button variant="ghost" size="sm" onClick={logout} className="gap-2 font-medium text-red-600 hover:text-red-700 hover:bg-red-50">
          <LogOut className="w-4 h-4" /> Logout
        </Button>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background font-sans text-foreground selection:bg-primary/10">

      {/* Header */}
      <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/95 backdrop-blur-xl shadow-sm">
        <div className="container flex h-20 items-center justify-between px-4 md:px-8 max-w-7xl mx-auto">
          <Link to={user ? (user.role === 'teacher' || user.role === 'admin' ? '/analytics' : '/parent-portal') : '/login'} className="flex items-center gap-3 hover:opacity-90 transition-opacity">
            <Logo />
            <div className="flex flex-col">
              <span className="text-2xl font-serif font-bold tracking-tight text-primary leading-none mt-1">Agnel Guardian</span>
              <span className="text-[10px] uppercase font-bold tracking-widest text-muted-foreground">Information Portal</span>
            </div>
          </Link>

          {/* Conditional Navigation */}
          {user && <NavLinks />}
        </div>
      </header>

      <main className="container max-w-7xl mx-auto px-4 py-8 md:py-12">
        <Routes>
          {/* Public Route */}
          <Route path="/login" element={user ? <Navigate to="/" replace /> : <Login />} />

          {/* Protected Teacher/Admin Routes */}
          <Route path="/extraction" element={<ProtectedRoute allowedRoles={['admin', 'teacher']}><ExtractionView /></ProtectedRoute>} />
          <Route path="/analytics" element={<ProtectedRoute allowedRoles={['admin', 'teacher']}><AnalyticsView /></ProtectedRoute>} />
          <Route path="/mailbox" element={<ProtectedRoute allowedRoles={['admin', 'teacher']}><TeacherMailbox /></ProtectedRoute>} />
          <Route path="/admin/users" element={<ProtectedRoute allowedRoles={['admin']}><AdminUsers /></ProtectedRoute>} />

          {/* Protected Parent/Student Routes */}
          <Route path="/parent-portal" element={<ProtectedRoute allowedRoles={['parent', 'student']}><ParentDashboard /></ProtectedRoute>} />

          {/* Shared Protected Route */}
          <Route path="/profile" element={<ProtectedRoute><ProfileManagement /></ProtectedRoute>} />

          {/* Default Fallback */}
          <Route path="*" element={
            <Navigate to={user ? (user.role === 'admin' || user.role === 'teacher' ? "/analytics" : "/parent-portal") : "/login"} replace />
          } />
        </Routes>
      </main>

    </div>
  )
}

export default App
