import 'dart:io';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:video_player/video_player.dart';
import '../theme/app_theme.dart';

class VideoReplayCard extends StatefulWidget {
  final File? videoFile;
  final String? videoUrl;
  final String? sampleName;
  final String? trueDirection;
  final String predictedDirection;

  const VideoReplayCard({
    super.key,
    this.videoFile,
    this.videoUrl,
    this.sampleName,
    this.trueDirection,
    required this.predictedDirection,
  });

  @override
  State<VideoReplayCard> createState() => _VideoReplayCardState();
}

class _VideoReplayCardState extends State<VideoReplayCard> {
  VideoPlayerController? _controller;
  bool _isInitialized = false;
  bool _hasError = false;

  @override
  void initState() {
    super.initState();
    _initPlayer();
  }

  Future<void> _initPlayer() async {
    try {
      if (widget.videoFile != null) {
        _controller = VideoPlayerController.file(widget.videoFile!);
      } else if (widget.videoUrl != null && widget.videoUrl!.isNotEmpty) {
        _controller = VideoPlayerController.networkUrl(Uri.parse(widget.videoUrl!));
      }

      if (_controller != null) {
        await _controller!.initialize();
        _controller!.setLooping(true);
        if (mounted) {
          setState(() {
            _isInitialized = true;
          });
        }
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _hasError = true;
        });
      }
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_hasError || (_controller == null && !_isInitialized)) {
      return const SizedBox.shrink();
    }

    final bool isMatch = widget.trueDirection != null &&
        widget.trueDirection!.toLowerCase() == widget.predictedDirection.toLowerCase();

    return Container(
      decoration: BoxDecoration(
        color: AppTheme.bgSurface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppTheme.borderSubtle),
      ),
      padding: const EdgeInsets.all(14),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Row(
                children: [
                  const Icon(Icons.play_circle_outline_rounded, size: 18, color: AppTheme.accentCyan),
                  const SizedBox(width: 8),
                  Text(
                    'VIDEO REPLAY & VERIFICATION',
                    style: GoogleFonts.outfit(
                      fontSize: 12,
                      fontWeight: FontWeight.w700,
                      letterSpacing: 1.0,
                      color: AppTheme.accentCyan,
                    ),
                  ),
                ],
              ),
              if (widget.trueDirection != null) ...[
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                  decoration: BoxDecoration(
                    color: (isMatch ? AppTheme.accentEmerald : AppTheme.accentAmber).withValues(alpha: 0.15),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(
                      color: (isMatch ? AppTheme.accentEmerald : AppTheme.accentAmber).withValues(alpha: 0.4),
                    ),
                  ),
                  child: Row(
                    children: [
                      Icon(
                        isMatch ? Icons.check_circle_rounded : Icons.info_outline_rounded,
                        size: 13,
                        color: isMatch ? AppTheme.accentEmerald : AppTheme.accentAmber,
                      ),
                      const SizedBox(width: 4),
                      Text(
                        'Actual: ${widget.trueDirection}',
                        style: GoogleFonts.robotoMono(
                          fontSize: 11,
                          fontWeight: FontWeight.w600,
                          color: isMatch ? AppTheme.accentEmerald : AppTheme.accentAmber,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ],
          ),
          const SizedBox(height: 10),

          // Video Container
          Container(
            height: 220,
            decoration: BoxDecoration(
              color: Colors.black,
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: AppTheme.borderSubtle),
            ),
            clipBehavior: Clip.antiAlias,
            child: _isInitialized && _controller != null
                ? Stack(
                    alignment: Alignment.center,
                    children: [
                      AspectRatio(
                        aspectRatio: _controller!.value.aspectRatio,
                        child: VideoPlayer(_controller!),
                      ),
                      // Play/Pause Overlay Button
                      GestureDetector(
                        onTap: () {
                          setState(() {
                            _controller!.value.isPlaying ? _controller!.pause() : _controller!.play();
                          });
                        },
                        child: Container(
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            color: Colors.black.withValues(alpha: _controller!.value.isPlaying ? 0.0 : 0.5),
                            shape: BoxShape.circle,
                          ),
                          child: Icon(
                            _controller!.value.isPlaying ? Icons.pause_rounded : Icons.play_arrow_rounded,
                            size: 36,
                            color: _controller!.value.isPlaying ? Colors.transparent : Colors.white,
                          ),
                        ),
                      ),
                      // Progress Bar at Bottom
                      Positioned(
                        bottom: 0,
                        left: 0,
                        right: 0,
                        child: VideoProgressIndicator(
                          _controller!,
                          allowScrubbing: true,
                          colors: const VideoProgressColors(
                            playedColor: AppTheme.accentEmerald,
                            bufferedColor: Colors.white24,
                            backgroundColor: Colors.black54,
                          ),
                        ),
                      ),
                    ],
                  )
                : const Center(
                    child: CircularProgressIndicator(strokeWidth: 2, color: AppTheme.accentCyan),
                  ),
          ),
          const SizedBox(height: 8),

          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                widget.sampleName ?? 'Uploaded Penalty Clip',
                style: GoogleFonts.inter(fontSize: 11, color: AppTheme.textMuted),
              ),
              Row(
                children: [
                  Text(
                    'Tap to Play / Pause',
                    style: GoogleFonts.inter(fontSize: 10, color: AppTheme.textSubtle),
                  ),
                  const SizedBox(width: 6),
                  const Icon(Icons.touch_app_rounded, size: 12, color: AppTheme.textSubtle),
                ],
              ),
            ],
          ),
        ],
      ),
    );
  }
}
