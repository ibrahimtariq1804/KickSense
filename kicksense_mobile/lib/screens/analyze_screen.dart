import 'dart:io';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';
import 'package:video_player/video_player.dart';
import '../models/prediction_result.dart';
import '../services/api_service.dart';
import '../theme/app_theme.dart';
import 'results_screen.dart';

class AnalyzeScreen extends StatefulWidget {
  final File? initialVideoFile;

  const AnalyzeScreen({super.key, this.initialVideoFile});

  @override
  State<AnalyzeScreen> createState() => _AnalyzeScreenState();
}

class _AnalyzeScreenState extends State<AnalyzeScreen> {
  final ImagePicker _picker = ImagePicker();
  File? _selectedFile;
  VideoPlayerController? _videoController;
  bool _isLoading = false;
  String _loadingStatus = '';
  String _currentServerUrl = '';

  @override
  void initState() {
    super.initState();
    _loadServerUrl();
    if (widget.initialVideoFile != null) {
      _setVideoFile(widget.initialVideoFile!);
    }
  }

  Future<void> _loadServerUrl() async {
    final url = await ApiService.getBaseUrl();
    setState(() {
      _currentServerUrl = url;
    });
  }

  Future<void> _pickVideo(ImageSource source) async {
    try {
      final XFile? picked = await _picker.pickVideo(
        source: source,
        maxDuration: const Duration(seconds: 30),
      );
      if (picked != null) {
        _setVideoFile(File(picked.path));
      }
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Could not access video: $e')),
      );
    }
  }

  void _setVideoFile(File file) {
    _videoController?.dispose();
    _selectedFile = file;

    _videoController = VideoPlayerController.file(file)
      ..initialize().then((_) {
        setState(() {});
        _videoController?.setLooping(true);
        _videoController?.play();
      });

    setState(() {});
  }

  Future<void> _runAnalysis() async {
    if (_selectedFile == null || _isLoading) return;

    setState(() {
      _isLoading = true;
      _loadingStatus = 'Uploading penalty kick video...';
    });

    try {
      setState(() {
        _loadingStatus = 'Detecting foot-ball contact & extracting kinematics...';
      });

      final PredictionResult result = await ApiService.predictVideo(_selectedFile!);

      if (!mounted) return;
      setState(() {
        _isLoading = false;
      });

      Navigator.of(context).push(
        MaterialPageRoute(
          builder: (_) => ResultsScreen(
            result: result,
            videoFile: _selectedFile,
            sampleName: 'Uploaded Penalty Video',
          ),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _isLoading = false;
      });
      _showErrorDialog(e.toString().replaceAll('Exception: ', ''));
    }
  }

  void _showErrorDialog(String error) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: AppTheme.bgCard,
        title: Text(
          'Analysis Failed',
          style: GoogleFonts.outfit(fontWeight: FontWeight.w700, color: AppTheme.accentRose),
        ),
        content: Text(
          error,
          style: GoogleFonts.inter(color: AppTheme.textMuted),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(),
            child: const Text('OK'),
          ),
          TextButton(
            onPressed: () {
              Navigator.of(ctx).pop();
              _showServerConfigDialog();
            },
            child: const Text('Change Server IP'),
          ),
        ],
      ),
    );
  }

  void _showServerConfigDialog() {
    final controller = TextEditingController(text: _currentServerUrl);
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: AppTheme.bgCard,
        title: Text(
          'Backend Server Configuration',
          style: GoogleFonts.outfit(fontWeight: FontWeight.w700),
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Specify your PC local IP or emulator address:',
              style: GoogleFonts.inter(fontSize: 13, color: AppTheme.textMuted),
            ),
            const SizedBox(height: 10),
            TextField(
              controller: controller,
              decoration: InputDecoration(
                filled: true,
                fillColor: AppTheme.bgSurface,
                hintText: 'http://192.168.1.5:8000',
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(10),
                  borderSide: const BorderSide(color: AppTheme.borderSubtle),
                ),
              ),
              style: GoogleFonts.robotoMono(fontSize: 13),
            ),
            const SizedBox(height: 10),
            Text(
              '• Android Emulator: http://10.0.2.2:8000\n• Physical Phone: http://<PC_LOCAL_IP>:8000\n• Windows/Web: http://127.0.0.1:8000',
              style: GoogleFonts.inter(fontSize: 11, color: AppTheme.textSubtle),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () async {
              await ApiService.setBaseUrl(controller.text.trim());
              _loadServerUrl();
              if (ctx.mounted) Navigator.of(ctx).pop();
            },
            child: const Text('Save'),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _videoController?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Analyze Penalty Kick'),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings_outlined),
            tooltip: 'Server Settings',
            onPressed: _showServerConfigDialog,
          ),
        ],
      ),
      body: Stack(
        children: [
          SingleChildScrollView(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // Server Status Pill
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                  decoration: BoxDecoration(
                    color: AppTheme.bgSurface,
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: AppTheme.borderSubtle),
                  ),
                  child: Row(
                    children: [
                      Container(
                        width: 8,
                        height: 8,
                        decoration: const BoxDecoration(
                          color: AppTheme.accentEmerald,
                          shape: BoxShape.circle,
                        ),
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          'Backend: $_currentServerUrl',
                          style: GoogleFonts.robotoMono(
                            fontSize: 11,
                            color: AppTheme.textMuted,
                          ),
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                      GestureDetector(
                        onTap: _showServerConfigDialog,
                        child: Text(
                          'Change',
                          style: GoogleFonts.inter(
                            fontSize: 12,
                            fontWeight: FontWeight.w600,
                            color: AppTheme.accentCyan,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 16),

                // Video Preview or Drop Box
                Container(
                  height: 280,
                  decoration: BoxDecoration(
                    color: Colors.black.withValues(alpha: 0.6),
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(
                      color: _selectedFile != null ? AppTheme.accentEmerald : AppTheme.borderBright,
                      width: 1.5,
                    ),
                  ),
                  clipBehavior: Clip.antiAlias,
                  child: _selectedFile != null && _videoController != null && _videoController!.value.isInitialized
                      ? Stack(
                          alignment: Alignment.center,
                          children: [
                            AspectRatio(
                              aspectRatio: _videoController!.value.aspectRatio,
                              child: VideoPlayer(_videoController!),
                            ),
                            Positioned(
                              bottom: 12,
                              right: 12,
                              child: FloatingActionButton.small(
                                backgroundColor: Colors.black54,
                                child: Icon(
                                  _videoController!.value.isPlaying ? Icons.pause : Icons.play_arrow,
                                  color: Colors.white,
                                ),
                                onPressed: () {
                                  setState(() {
                                    _videoController!.value.isPlaying
                                        ? _videoController!.pause()
                                        : _videoController!.play();
                                  });
                                },
                              ),
                            ),
                          ],
                        )
                      : Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Container(
                              width: 64,
                              height: 64,
                              decoration: BoxDecoration(
                                color: AppTheme.accentEmerald.withValues(alpha: 0.15),
                                shape: BoxShape.circle,
                              ),
                              child: const Icon(
                                Icons.sports_soccer_rounded,
                                size: 36,
                                color: AppTheme.accentEmerald,
                              ),
                            ),
                            const SizedBox(height: 16),
                            Text(
                              'Select or Record Penalty Clip',
                              style: GoogleFonts.outfit(
                                fontSize: 18,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                            const SizedBox(height: 6),
                            Text(
                              'Supports MP4 or MOV clips showing run-up up to kick moment',
                              textAlign: TextAlign.center,
                              style: GoogleFonts.inter(
                                fontSize: 12,
                                color: AppTheme.textMuted,
                              ),
                            ),
                          ],
                        ),
                ),
                const SizedBox(height: 16),

                // Video Capture / Pick Action Buttons
                Row(
                  children: [
                    Expanded(
                      child: OutlinedButton.icon(
                        onPressed: () => _pickVideo(ImageSource.camera),
                        icon: const Icon(Icons.videocam_rounded, color: AppTheme.accentCyan),
                        label: const Text('Record Video'),
                        style: OutlinedButton.styleFrom(
                          foregroundColor: Colors.white,
                          side: const BorderSide(color: AppTheme.borderBright),
                          padding: const EdgeInsets.symmetric(vertical: 14),
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                        ),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: OutlinedButton.icon(
                        onPressed: () => _pickVideo(ImageSource.gallery),
                        icon: const Icon(Icons.photo_library_rounded, color: AppTheme.accentEmerald),
                        label: const Text('From Gallery'),
                        style: OutlinedButton.styleFrom(
                          foregroundColor: Colors.white,
                          side: const BorderSide(color: AppTheme.borderBright),
                          padding: const EdgeInsets.symmetric(vertical: 14),
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 24),

                // Info Box
                Container(
                  padding: const EdgeInsets.all(14),
                  decoration: BoxDecoration(
                    color: AppTheme.accentCyan.withValues(alpha: 0.08),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: AppTheme.accentCyan.withValues(alpha: 0.25)),
                  ),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Icon(Icons.info_outline_rounded, size: 20, color: AppTheme.accentCyan),
                      const SizedBox(width: 10),
                      Expanded(
                        child: Text(
                          'Our smart contact detector automatically identifies the foot-ball strike moment (t ≤ strike), cuts off post-kick follow through, and standardizes to 26 kinematic frames.',
                          style: GoogleFonts.inter(fontSize: 12, color: Color(0xFFBAE6FD), height: 1.4),
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 24),

                // Main Analyze Button
                ElevatedButton.icon(
                  onPressed: _selectedFile != null && !_isLoading ? _runAnalysis : null,
                  icon: const Icon(Icons.bolt_rounded),
                  label: const Text('ANALYZE PENALTY KICK'),
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 18),
                    backgroundColor: AppTheme.accentEmerald,
                    disabledBackgroundColor: Colors.white12,
                  ),
                ),
                const SizedBox(height: 20),
              ],
            ),
          ),

          // Loading Overlay
          if (_isLoading)
            Container(
              color: Colors.black.withValues(alpha: 0.8),
              alignment: Alignment.center,
              padding: const EdgeInsets.all(32),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const SizedBox(
                    width: 56,
                    height: 56,
                    child: CircularProgressIndicator(
                      color: AppTheme.accentEmerald,
                      strokeWidth: 3.5,
                    ),
                  ),
                  const SizedBox(height: 24),
                  Text(
                    'Analyzing Kick Kinematics...',
                    style: GoogleFonts.outfit(
                      fontSize: 20,
                      fontWeight: FontWeight.w700,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    _loadingStatus,
                    textAlign: TextAlign.center,
                    style: GoogleFonts.inter(
                      fontSize: 13,
                      color: AppTheme.textMuted,
                    ),
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }
}
