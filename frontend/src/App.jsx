import React, { useState, useEffect } from 'react'
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
import { Loader2, ChevronDown, GraduationCap, ShieldCheck, RefreshCw, Trash2, LineChart, Upload, Plus, Users } from 'lucide-react'

// Elegant Logo Component
const Logo = () => (
  <div className="flex items-center gap-2">
    <div className="flex h-10 w-10 items-center justify-center rounded-sm bg-primary text-primary-foreground shadow-sm">
      <GraduationCap className="h-6 w-6" />
    </div>
  </div>
)

function App() {
  // --- STATE ---
  const fileInputRef = React.useRef(null)

  const [files, setFiles] = useState([]) // Array of File objects
  const [results, setResults] = useState({}) // Map: index -> { status: 'pending'|'loading'|'success'|'error', data: ..., error: ... }
  const [selectedIndex, setSelectedIndex] = useState(0) // Index of currently viewed file
  const [viewMode, setViewMode] = useState('extraction') // 'extraction' | 'analytics'

  const [preview, setPreview] = useState(null) // Base64 of CURRENT viewed file
  const [mode, setMode] = useState('pta_free')
  const [template, setTemplate] = useState('')
  const [debug, setDebug] = useState(false)
  const [department, setDepartment] = useState('CSE')
  const [className, setClassName] = useState('')

  const [batchProcessing, setBatchProcessing] = useState(false)
  const [aiSummary, setAiSummary] = useState(null)
  const [isGeneratingSummary, setIsGeneratingSummary] = useState(false)

  // --- HANDLER: File Selection ---
  const handleFilesSelected = (newFiles) => {
    // Append or Replace? Let's Replace for simplicity to start (clear old).
    // Or Append? "Inbox" implies appending. Let's start with Replace/Set.
    // If user drops new files, we reset.
    setFiles(newFiles)

    // Initialize results state
    const initialResults = {}
    newFiles.forEach((_, idx) => {
      initialResults[idx] = { status: 'pending' }
    })
    setResults(initialResults)

    setSelectedIndex(0)
    setAiSummary(null) // Clear previous summary
  }

  // --- HANDLER: Append Files ---
  const handleAppendFiles = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const appended = Array.from(e.target.files)

      // Calculate new starting index
      const startIndex = files.length

      setFiles(prev => [...prev, ...appended])

      setResults(prev => {
        const next = { ...prev }
        appended.forEach((_, i) => {
          next[startIndex + i] = { status: 'pending' }
        })
        return next
      })

      // Reset input so same file selection triggers change again if needed
      e.target.value = null
    }
  }

  // --- EFFECT: Load Preview when Selected Index changes ---
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


  // --- LOGIC: Helper to process ONE file ---
  const processSingleFile = async (file, index) => {
    try {
      // 1. Read File to Base64 (Promise wrapper)
      const toBase64 = (f) => new Promise((resolve, reject) => {
        const r = new FileReader()
        r.readAsDataURL(f)
        r.onload = () => resolve(r.result)
        r.onerror = error => reject(error)
      })
      const base64 = await toBase64(file)

      // 2. Prepare Payload
      const url =
        mode === "pta"
          ? "/ocr/pta"
          : mode === "pta_free"
            ? "/ocr/pta_free"
            : "/ocr/auto"

      const payload = {
        imageBase64: base64, // The API expects raw field name 'imageBase64'
        template: template || null,
        debug: debug,
        department: department,
        class_name: className
      }

      // 3. Update Status -> Loading
      setResults(prev => ({
        ...prev,
        [index]: { ...prev[index], status: 'loading' }
      }))

      // 4. API Call
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`)
      }

      const data = await response.json()

      // 5. Update Status -> Success
      setResults(prev => ({
        ...prev,
        [index]: { status: 'success', data: data }
      }))

      return data  // Return the data so batch can collect comments

    } catch (err) {
      console.error(`Error processing file ${index}:`, err)

      // 6. Update Status -> Error
      setResults(prev => ({
        ...prev,
        [index]: { status: 'error', error: err.message || "Failed" }
      }))
      return null
    }
  }

  // --- LOGIC: Batch Loop with Auto-Summary ---
  const runBatchOCR = async () => {
    setBatchProcessing(true)
    setAiSummary(null)

    // Ordered storage for comments
    const commentsByIndex = new Array(files.length).fill(null);

    // Create a queue of pending file indices
    const queue = files.map((file, index) => ({ file, index }));
    const activeWorkers = [];
    const CONCURRENCY_LIMIT = 5;

    const worker = async () => {
      while (queue.length > 0) {
        const { file, index } = queue.shift();

        if (results[index]?.status === 'success') {
          // Already processed, get comment from existing result
          if (results[index].data?.comments) {
            commentsByIndex[index] = results[index].data.comments;
          }
          continue;
        }

        const data = await processSingleFile(file, index);
        if (data?.comments) {
          commentsByIndex[index] = data.comments;
        }
      }
    };

    // Start workers
    for (let i = 0; i < CONCURRENCY_LIMIT; i++) {
      activeWorkers.push(worker());
    }

    await Promise.all(activeWorkers);

    // Reconstruct collectedComments in order
    const collectedComments = commentsByIndex.filter(c => c !== null);

    setBatchProcessing(false)

    console.log("Collected comments:", collectedComments)

    // Now call OpenAI if we have comments
    if (collectedComments.length > 0) {
      setIsGeneratingSummary(true)
      try {
        const response = await fetch('/analysis/summarize_ai', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ comments: collectedComments })
        })
        const data = await response.json()
        console.log("Summary data:", data)
        setAiSummary(data)
      } catch (e) {
        console.error("Summary error:", e)
        setAiSummary({ error: 'Failed to generate summary', details: e.message })
      } finally {
        setIsGeneratingSummary(false)
      }
    } else {
      console.log("No comments to summarize")
    }
  }

  const clearAll = () => {
    setFiles([])
    setResults({})
    setSelectedIndex(0)
    setPreview(null)
    setAiSummary(null)
  }

  // Computed for UI
  const currentResult = results[selectedIndex] || {}
  const hasFiles = files.length > 0

  return (
    <div className="min-h-screen bg-background font-sans text-foreground selection:bg-primary/10">

      {/* Header */}
      <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/80 backdrop-blur-md">
        <div className="container flex h-20 items-center justify-between px-4 md:px-8 max-w-7xl mx-auto">
          <div className="flex items-center gap-4">
            <Logo />
            <div className="flex flex-col">
              <span className="text-xl font-serif font-bold tracking-tight text-primary">Agnel Guardian</span>
            </div>
          </div>

          <div className="flex items-center bg-slate-100 p-1 rounded-lg">
            <Button
              variant={viewMode === 'extraction' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setViewMode('extraction')}
              className="gap-2"
            >
              <Upload className="w-4 h-4" /> Extraction
            </Button>
            <Button
              variant={viewMode === 'analytics' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setViewMode('analytics')}
              className="gap-2"
            >
              <LineChart className="w-4 h-4" /> Analytics
            </Button>
            <Button
              variant={viewMode === 'parent' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setViewMode('parent')}
              className="gap-2"
            >
              <Users className="w-4 h-4" /> Parent Portal
            </Button>
          </div>
        </div>
      </header>

      <main className="container max-w-7xl mx-auto px-4 py-8 md:py-12 space-y-8">

        {viewMode === 'parent' ? (
          <ParentDashboard />
        ) : viewMode === 'analytics' ? (
          <AnalyticsView />
        ) : !hasFiles ? (
          // --- VIEW 1: EMPTY STATE (HERO UPLOAD) ---
          <div className="max-w-3xl mx-auto space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
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



            <div className="flex justify-center gap-4 text-sm text-muted-foreground">
              <div className="flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-primary" /> Secure Processing
              </div>
              <div className="flex items-center gap-2">
                <RefreshCw className="w-4 h-4 text-primary" /> Auto-Scaling Logic
              </div>
            </div>
          </div>
        ) : (
          // --- VIEW 2: WORKSPACE (INBOX GRID) ---
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start animate-in fade-in duration-500">

            {/* --- LEFT COLUMN: CONFIG + INBOX --- */}
            <div className="lg:col-span-4 space-y-6 flex flex-col h-[calc(100vh-140px)] sticky top-24">

              {/* Action Bar */}
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



              {/* Sidebar Inbox List (Scrollable) */}
              <div className="flex-1 min-h-0">
                <SidebarList
                  files={files}
                  activeIndex={selectedIndex}
                  onSelect={setSelectedIndex}
                  results={results}
                />
              </div>

            </div>


            {/* --- RIGHT COLUMN: RESULT DETAIL --- */}
            <div className="lg:col-span-8 space-y-6">
              {/* Detail View for Selected File */}
              <div className="min-h-[600px]">
                <ResultsDisplay
                  data={currentResult.data || {}}
                  previewUrl={preview}
                // If we have an error, we can pass it or handle it in ResultsDisplay? 
                // ResultsDisplay expects 'data'. Let's handle 'error' roughly here for now.
                />

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

                {/* AI Summary - Inline after results */}
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

      </main>


    </div>
  )
}

export default App
