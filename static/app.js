/**
 * KickSense Web Application Logic
 * High-Aesthetics UI, Drag-and-Drop, Stadium Goal Visualizer & Biomechanics Telemetry
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const dropzone = document.getElementById('dropzone');
  const videoFileInput = document.getElementById('videoFileInput');
  const dropzonePrompt = document.getElementById('dropzonePrompt');
  const previewWrapper = document.getElementById('previewWrapper');
  const videoPlayer = document.getElementById('videoPlayer');
  
  const analyzeBtn = document.getElementById('analyzeBtn');
  const btnSpinner = document.getElementById('btnSpinner');
  const btnText = document.getElementById('btnText');
  const videoResolutionBadge = document.getElementById('videoResolutionBadge');
  const inferenceStatusBadge = document.getElementById('inferenceStatusBadge');

  const samplePillsContainer = document.getElementById('samplePillsContainer');
  
  const skeletonSection = document.getElementById('skeletonSection');
  const skeletonImg = document.getElementById('skeletonImg');
  const contactFrameTag = document.getElementById('contactFrameTag');
  
  const emptyState = document.getElementById('emptyState');
  const resultsContent = document.getElementById('resultsContent');

  // Goal Visualizer Zones
  const zoneLeft = document.getElementById('zoneLeft');
  const zoneCenter = document.getElementById('zoneCenter');
  const zoneRight = document.getElementById('zoneRight');
  const zoneLeftProb = document.getElementById('zoneLeftProb');
  const zoneCenterProb = document.getElementById('zoneCenterProb');
  const zoneRightProb = document.getElementById('zoneRightProb');

  // Primary Metrics
  const resArrow = document.getElementById('resArrow');
  const resDirection = document.getElementById('resDirection');
  const resConfidence = document.getElementById('resConfidence');
  const resConfidenceFill = document.getElementById('resConfidenceFill');
  const resPower = document.getElementById('resPower');
  const resPowerTag = document.getElementById('resPowerTag');
  const resPowerFill = document.getElementById('resPowerFill');

  // Probability Bars
  const barLeft = document.getElementById('barLeft');
  const barCenter = document.getElementById('barCenter');
  const barRight = document.getElementById('barRight');
  const pctLeft = document.getElementById('pctLeft');
  const pctCenter = document.getElementById('pctCenter');
  const pctRight = document.getElementById('pctRight');

  // Telemetry
  const tTorso = document.getElementById('tTorso');
  const tTorsoHint = document.getElementById('tTorsoHint');
  const tPelvis = document.getElementById('tPelvis');
  const tPelvisHint = document.getElementById('tPelvisHint');
  const tSpeed = document.getElementById('tSpeed');
  const tOffset = document.getElementById('tOffset');
  const inferenceTimeTag = document.getElementById('inferenceTimeTag');

  // Modal
  const modalBackdrop = document.getElementById('modalBackdrop');
  const openDocsBtn = document.getElementById('openDocsBtn');
  const closeDocsBtn = document.getElementById('closeDocsBtn');

  // Application State
  let selectedFile = null;
  let activeSampleId = null;
  let isProcessing = false;

  // 1. Initialize Sample Clips
  fetchSampleClips();

  // 2. Setup Modal Handlers
  if (openDocsBtn && modalBackdrop && closeDocsBtn) {
    openDocsBtn.addEventListener('click', () => {
      modalBackdrop.hidden = false;
    });
    closeDocsBtn.addEventListener('click', () => {
      modalBackdrop.hidden = true;
    });
    modalBackdrop.addEventListener('click', (e) => {
      if (e.target === modalBackdrop) {
        modalBackdrop.hidden = true;
      }
    });
  }

  // 3. Dropzone Interactions
  dropzone.addEventListener('click', (e) => {
    // If click was on video controls, don't trigger file dialog
    if (e.target === videoPlayer || isProcessing) return;
    videoFileInput.click();
  });

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('drag-over');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('drag-over');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFileSelected(files[0]);
    }
  });

  videoFileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  // 4. File Selection Handler
  function handleFileSelected(file) {
    const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
    const valid = ['.mp4', '.mov', '.avi'];
    if (!valid.includes(ext) && !file.type.startsWith('video/')) {
      alert(`Unsupported file format. Please upload an MP4, MOV, or AVI video.`);
      return;
    }

    selectedFile = file;
    activeSampleId = null;
    clearSamplePillActive();

    // Load preview
    const videoURL = URL.createObjectURL(file);
    videoPlayer.src = videoURL;
    videoPlayer.load();

    previewWrapper.hidden = false;
    dropzonePrompt.hidden = true;
    analyzeBtn.disabled = false;
    videoResolutionBadge.textContent = file.name;
    videoResolutionBadge.classList.add('badge-live');
    inferenceStatusBadge.textContent = 'Ready to Analyze';
    inferenceStatusBadge.classList.remove('active');

    videoPlayer.onloadedmetadata = () => {
      const duration = videoPlayer.duration.toFixed(1);
      videoResolutionBadge.textContent = `${videoPlayer.videoWidth}x${videoPlayer.videoHeight} · ${duration}s`;
    };
  }

  // 5. Sample Clips Fetching & Click Handler
  async function fetchSampleClips() {
    try {
      const res = await fetch('/api/sample-clips');
      if (!res.ok) throw new Error('Failed to fetch samples');
      const data = await res.json();
      
      samplePillsContainer.innerHTML = '';
      if (!data.samples || data.samples.length === 0) {
        samplePillsContainer.innerHTML = '<span class="loading-pills">No sample clips found.</span>';
        return;
      }

      data.samples.forEach(sample => {
        const btn = document.createElement('button');
        btn.className = 'sample-pill-btn';
        btn.dataset.id = sample.id;
        btn.innerHTML = `
          <span>⚽ ${sample.name}</span>
          <span style="opacity: 0.6; font-size: 0.7rem;">(${sample.true_direction})</span>
        `;
        btn.addEventListener('click', () => loadSampleKick(sample, btn));
        samplePillsContainer.appendChild(btn);
      });
    } catch (err) {
      console.warn('Sample clips not loaded:', err);
      samplePillsContainer.innerHTML = '<span class="loading-pills">Sample clips unavailable</span>';
    }
  }

  function clearSamplePillActive() {
    document.querySelectorAll('.sample-pill-btn').forEach(btn => btn.classList.remove('active'));
  }

  function loadSampleKick(sample, btnElement) {
    if (isProcessing) return;
    
    clearSamplePillActive();
    btnElement.classList.add('active');
    activeSampleId = sample.id;
    selectedFile = null;

    // Set video player to stream sample clip
    const videoUrl = `/api/sample-video/${sample.id}`;
    videoPlayer.src = videoUrl;
    videoPlayer.load();
    previewWrapper.hidden = false;
    dropzonePrompt.hidden = true;
    analyzeBtn.disabled = false;

    videoResolutionBadge.textContent = `Sample: ${sample.id}.mp4`;
    videoResolutionBadge.classList.add('badge-live');
    inferenceStatusBadge.textContent = 'Sample Loaded';

    // Trigger analysis directly for convenient 1-click evaluation
    triggerAnalysis();
  }

  // 6. Action Button Click
  analyzeBtn.addEventListener('click', () => {
    triggerAnalysis();
  });

  async function triggerAnalysis() {
    if (isProcessing) return;
    if (!selectedFile && !activeSampleId) return;

    setProcessingState(true);

    try {
      let response;
      if (activeSampleId) {
        // Sample kick route
        response = await fetch(`/api/predict-sample?sample_id=${encodeURIComponent(activeSampleId)}`, {
          method: 'POST'
        });
      } else {
        // Uploaded file route
        const formData = new FormData();
        formData.append('file', selectedFile);
        response = await fetch('/predict', {
          method: 'POST',
          body: formData
        });
      }

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({ detail: 'Analysis failed' }));
        throw new Error(errJson.detail || `Server responded with ${response.status}`);
      }

      const data = await response.json();
      renderPredictionResults(data);

    } catch (err) {
      console.error('Prediction error:', err);
      alert(`KickSense Analysis Error: ${err.message}`);
      inferenceStatusBadge.textContent = 'Analysis Failed';
      inferenceStatusBadge.classList.remove('active');
    } finally {
      setProcessingState(false);
    }
  }

  function setProcessingState(processing) {
    isProcessing = processing;
    analyzeBtn.disabled = processing;
    btnSpinner.hidden = !processing;
    btnText.textContent = processing ? 'Scanning Kinematics...' : 'Analyze Penalty Kick';

    if (processing) {
      inferenceStatusBadge.textContent = 'Extracting Poses & Contact...';
      inferenceStatusBadge.classList.add('active');
    }
  }

  // 7. Render Prediction Results & Telemetry
  function renderPredictionResults(data) {
    // 1. Hide empty state, show results
    emptyState.hidden = true;
    resultsContent.hidden = false;
    inferenceStatusBadge.textContent = 'Inference Complete';
    inferenceStatusBadge.classList.add('active');

    // 2. Goal Net Visualizer Activation
    zoneLeft.classList.remove('active');
    zoneCenter.classList.remove('active');
    zoneRight.classList.remove('active');

    const dir = data.direction; // 'Left', 'Center', or 'Right'
    if (dir === 'Left') zoneLeft.classList.add('active');
    else if (dir === 'Center') zoneCenter.classList.add('active');
    else if (dir === 'Right') zoneRight.classList.add('active');

    zoneLeftProb.textContent = `${data.probabilities.Left.toFixed(0)}%`;
    zoneCenterProb.textContent = `${data.probabilities.Center.toFixed(0)}%`;
    zoneRightProb.textContent = `${data.probabilities.Right.toFixed(0)}%`;

    // 3. Direction Hero Card
    const arrowMap = { 'Left': '⬅️', 'Center': '⬆️', 'Right': '➡️' };
    resArrow.textContent = arrowMap[dir] || '⚽';
    resDirection.textContent = dir.toUpperCase();
    resConfidence.textContent = `${data.confidence.toFixed(1)}%`;
    resConfidenceFill.style.width = `${Math.min(100, Math.max(10, data.confidence))}%`;

    // 4. Power Rating Card
    const power = data.power; // 1.0 - 10.0
    resPower.textContent = power.toFixed(1);
    resPowerTag.textContent = data.power_category || 'Strike Power';
    const powerPct = ((power / 10.0) * 100).toFixed(0);
    resPowerFill.style.width = `${powerPct}%`;

    // 5. Probability Breakdown Bars
    barLeft.style.width = `${data.probabilities.Left}%`;
    barCenter.style.width = `${data.probabilities.Center}%`;
    barRight.style.width = `${data.probabilities.Right}%`;
    pctLeft.textContent = `${data.probabilities.Left.toFixed(1)}%`;
    pctCenter.textContent = `${data.probabilities.Center.toFixed(1)}%`;
    pctRight.textContent = `${data.probabilities.Right.toFixed(1)}%`;

    // 6. Biomechanical Telemetry
    if (data.biomechanics) {
      const bio = data.biomechanics;
      tTorso.textContent = `${bio.torso_lean_deg.toFixed(1)}°`;
      tTorsoHint.textContent = bio.torso_lean_label || 'Lean angle';

      tPelvis.textContent = `${bio.pelvis_angle_deg.toFixed(1)}°`;
      tPelvisHint.textContent = bio.pelvis_angle_label || 'Hip orientation';

      tSpeed.textContent = `${bio.kicking_ankle_peak_speed.toFixed(0)} px/s`;
      tOffset.textContent = `${bio.plant_to_ball_dist.toFixed(2)}`;
    }

    if (data.inference_time_ms) {
      inferenceTimeTag.textContent = `${data.inference_time_ms} ms`;
    }

    // 7. Skeletal Pose Snapshot
    if (data.strike_skeleton_base64) {
      skeletonImg.src = `data:image/jpeg;base64,${data.strike_skeleton_base64}`;
      contactFrameTag.textContent = `Strike Contact @ Frame #${data.strike_frame_idx}`;
      skeletonSection.hidden = false;
    } else {
      skeletonSection.hidden = true;
    }

    // Smooth scroll down slightly if on small mobile screen
    if (window.innerWidth < 980) {
      resultsContent.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }
});
