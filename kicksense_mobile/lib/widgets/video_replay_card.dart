import 'dart:io';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:video_player/video_player.dart';
import '../theme/app_theme.dart';

enum VideoProofMode { aiInputCut, fullOutcome }

class VideoReplayCard extends StatefulWidget {
  final File? videoFile;
  final String? videoUrl;
  final String? sampleName;
  final String? trueDirection;
  final String predictedDirection;
  final int strikeFrameIdx;
  final double fps;
  final int totalFrames;

  const VideoReplayCard({
    super.key,
    this.videoFile,
    this.videoUrl,
    this.sampleName,
    this.trueDirection,
    required this.predictedDirection,
    required this.strikeFrameIdx,
    required this.fps,
    required this.totalFrames,
  });

  @override
  State<VideoReplayCard> createState() => _VideoReplayCardState();
}

class _VideoReplayCardState extends State<VideoReplayCard> {
  VideoPlayerController? _controller;
  bool _isInitialized = false;
  bool _hasError = false;
  VideoProofMode _mode = VideoProofMode.aiInputCut;

  Duration get _strikeDuration {
    final double safeFps = widget.fps > 0 ? widget.fps : 30.0;
    final int ms = ((widget.strikeFrameIdx / safeFps) * 1000).round();
    return Duration(milliseconds: ms > 0 ? ms : 1500);
  }

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
        _controller!.addListener(_onPlaybackTick);
        if (mounted) {
          setState(() {
            _isInitialized = true;
          });
          // Auto play up to contact in AI input mode
          _controller!.play();
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

  void _onPlaybackTick() {
    if (!mounted || _controller == null) return;

    // In AI Input mode, pause exactly at the strike contact moment
    if (_mode == VideoProofMode.aiInputCut && _controller!.value.isPlaying) {
      if (_controller!.value.position >= _strikeDuration) {
        _controller!.pause();
        _controller!.seekTo(_strikeDuration);
        if (mounted) setState(() {});
      }
    }
  }

  void _setMode(VideoProofMode newMode) {
    if (_mode == newMode) return;
    setState(() {
      _mode = newMode;
    });

    if (_controller != null && _isInitialized) {
      if (newMode == VideoProofMode.aiInputCut) {
        _controller!.seekTo(Duration.zero);
        _controller!.play();
      } else {
        _controller!.play();
      }
    }
  }

  void _jumpToImpact() {
    if (_controller != null && _isInitialized) {
      _controller!.pause();
      _controller!.seekTo(_strikeDuration);
      setState(() {});
    }
  }

  void _replayFromStart() {
    if (_controller != null && _isInitialized) {
      _controller!.seekTo(Duration.zero);
      _controller!.play();
      setState(() {});
    }
  }

  String _formatDuration(Duration d) {
    final int min = d.inMinutes;
    final int sec = d.inSeconds % 60;
    return '${min.toString().padLeft(2, '0')}:${sec.toString().padLeft(2, '0')}';
  }

  @override
  void dispose() {
    _controller?.removeListener(_onPlaybackTick);
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

    final Duration currentPos = _controller?.value.position ?? Duration.zero;
    final Duration totalDur = _controller?.value.duration ?? Duration.zero;
    final bool isAtOrPastContact = currentPos >= _strikeDuration;
    final double strikeSec = widget.strikeFrameIdx / (widget.fps > 0 ? widget.fps : 30.0);

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
          // Header Row with Zero-Overflow protection
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Expanded(
                child: Row(
                  children: [
                    const Icon(Icons.verified_user_rounded, size: 16, color: AppTheme.accentCyan),
                    const SizedBox(width: 6),
                    Expanded(
                      child: Text(
                        'SUPERVISOR PROOF & VIDEO',
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: GoogleFonts.outfit(
                          fontSize: 12,
                          fontWeight: FontWeight.w700,
                          letterSpacing: 0.8,
                          color: AppTheme.accentCyan,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              if (widget.trueDirection != null) ...[
                const SizedBox(width: 8),
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
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        isMatch ? Icons.check_circle_rounded : Icons.info_outline_rounded,
                        size: 12,
                        color: isMatch ? AppTheme.accentEmerald : AppTheme.accentAmber,
                      ),
                      const SizedBox(width: 4),
                      Text(
                        'Actual: ${widget.trueDirection}',
                        style: GoogleFonts.robotoMono(
                          fontSize: 10,
                          fontWeight: FontWeight.w700,
                          color: isMatch ? AppTheme.accentEmerald : AppTheme.accentAmber,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ],
          ),
          const SizedBox(height: 12),

          // Dual-Mode Selector Tabs (AI Input Clip vs Full Outcome Proof)
          Container(
            padding: const EdgeInsets.all(3),
            decoration: BoxDecoration(
              color: Colors.black.withValues(alpha: 0.4),
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: AppTheme.borderSubtle),
            ),
            child: Row(
              children: [
                Expanded(
                  child: GestureDetector(
                    onTap: () => _setMode(VideoProofMode.aiInputCut),
                    child: Container(
                      padding: const EdgeInsets.symmetric(vertical: 8),
                      decoration: BoxDecoration(
                        color: _mode == VideoProofMode.aiInputCut
                            ? AppTheme.accentEmerald.withValues(alpha: 0.25)
                            : Colors.transparent,
                        borderRadius: BorderRadius.circular(8),
                        border: _mode == VideoProofMode.aiInputCut
                            ? Border.all(color: AppTheme.accentEmerald.withValues(alpha: 0.6))
                            : null,
                      ),
                      alignment: Alignment.center,
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(
                            Icons.content_cut_rounded,
                            size: 13,
                            color: _mode == VideoProofMode.aiInputCut ? AppTheme.accentEmerald : AppTheme.textMuted,
                          ),
                          const SizedBox(width: 6),
                          Flexible(
                            child: Text(
                              'AI Input (Pre-Impact)',
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                              style: GoogleFonts.outfit(
                                fontSize: 11,
                                fontWeight: FontWeight.w700,
                                color: _mode == VideoProofMode.aiInputCut ? Colors.white : AppTheme.textMuted,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 4),
                Expanded(
                  child: GestureDetector(
                    onTap: () => _setMode(VideoProofMode.fullOutcome),
                    child: Container(
                      padding: const EdgeInsets.symmetric(vertical: 8),
                      decoration: BoxDecoration(
                        color: _mode == VideoProofMode.fullOutcome
                            ? AppTheme.accentCyan.withValues(alpha: 0.25)
                            : Colors.transparent,
                        borderRadius: BorderRadius.circular(8),
                        border: _mode == VideoProofMode.fullOutcome
                            ? Border.all(color: AppTheme.accentCyan.withValues(alpha: 0.6))
                            : null,
                      ),
                      alignment: Alignment.center,
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(
                            Icons.sports_soccer_rounded,
                            size: 13,
                            color: _mode == VideoProofMode.fullOutcome ? AppTheme.accentCyan : AppTheme.textMuted,
                          ),
                          const SizedBox(width: 6),
                          Flexible(
                            child: Text(
                              'Full Outcome (Proof)',
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                              style: GoogleFonts.outfit(
                                fontSize: 11,
                                fontWeight: FontWeight.w700,
                                color: _mode == VideoProofMode.fullOutcome ? Colors.white : AppTheme.textMuted,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 10),

          // Mode Description Banner
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
            decoration: BoxDecoration(
              color: (_mode == VideoProofMode.aiInputCut ? AppTheme.accentEmerald : AppTheme.accentCyan).withValues(alpha: 0.08),
              borderRadius: BorderRadius.circular(8),
              border: Border.all(
                color: (_mode == VideoProofMode.aiInputCut ? AppTheme.accentEmerald : AppTheme.accentCyan).withValues(alpha: 0.2),
              ),
            ),
            child: Row(
              children: [
                Icon(
                  _mode == VideoProofMode.aiInputCut ? Icons.lock_clock_rounded : Icons.check_circle_outline_rounded,
                  size: 14,
                  color: _mode == VideoProofMode.aiInputCut ? AppTheme.accentEmerald : AppTheme.accentCyan,
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    _mode == VideoProofMode.aiInputCut
                        ? 'Stops at impact (frame #${widget.strikeFrameIdx} / ${strikeSec.toStringAsFixed(2)}s). Shows model had 0 flight outcome leak.'
                        : 'Full clip replay. Proves whether predicted direction (${widget.predictedDirection}) matches where ball landed in goal.',
                    style: GoogleFonts.inter(fontSize: 10.5, color: Colors.white70),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 10),

          // Video Player Screen
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

                      // Impact contact badge overlay when paused at impact in AI Cut mode
                      if (_mode == VideoProofMode.aiInputCut && isAtOrPastContact)
                        Positioned(
                          top: 10,
                          child: Container(
                            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                            decoration: BoxDecoration(
                              color: AppTheme.accentEmerald.withValues(alpha: 0.9),
                              borderRadius: BorderRadius.circular(20),
                              boxShadow: [
                                BoxShadow(
                                  color: AppTheme.accentEmerald.withValues(alpha: 0.4),
                                  blurRadius: 10,
                                ),
                              ],
                            ),
                            child: Row(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                const Icon(Icons.bolt_rounded, size: 14, color: Colors.black),
                                const SizedBox(width: 4),
                                Text(
                                  'IMPACT CONTACT POINT (CUTOFF)',
                                  style: GoogleFonts.outfit(
                                    fontSize: 10,
                                    fontWeight: FontWeight.w800,
                                    color: Colors.black,
                                    letterSpacing: 0.5,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),

                      // Play/Pause Touch Overlay
                      GestureDetector(
                        onTap: () {
                          if (_controller!.value.isPlaying) {
                            _controller!.pause();
                          } else {
                            if (_mode == VideoProofMode.aiInputCut && isAtOrPastContact) {
                              _replayFromStart();
                            } else {
                              _controller!.play();
                            }
                          }
                          setState(() {});
                        },
                        child: Container(
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            color: Colors.black.withValues(alpha: _controller!.value.isPlaying ? 0.0 : 0.45),
                            shape: BoxShape.circle,
                          ),
                          child: Icon(
                            _controller!.value.isPlaying ? Icons.pause_rounded : Icons.play_arrow_rounded,
                            size: 38,
                            color: _controller!.value.isPlaying ? Colors.transparent : Colors.white,
                          ),
                        ),
                      ),

                      // Progress Bar Scrubber
                      Positioned(
                        bottom: 0,
                        left: 0,
                        right: 0,
                        child: VideoProgressIndicator(
                          _controller!,
                          allowScrubbing: true,
                          colors: VideoProgressColors(
                            playedColor: _mode == VideoProofMode.aiInputCut ? AppTheme.accentEmerald : AppTheme.accentCyan,
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

          // Player Control Actions & Timing Row
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Row(
                children: [
                  IconButton(
                    iconSize: 22,
                    padding: EdgeInsets.zero,
                    constraints: const BoxConstraints(),
                    icon: Icon(
                      _controller != null && _controller!.value.isPlaying
                          ? Icons.pause_circle_filled_rounded
                          : Icons.play_circle_fill_rounded,
                      color: AppTheme.accentEmerald,
                    ),
                    onPressed: () {
                      if (_controller == null) return;
                      if (_controller!.value.isPlaying) {
                        _controller!.pause();
                      } else {
                        if (_mode == VideoProofMode.aiInputCut && isAtOrPastContact) {
                          _replayFromStart();
                        } else {
                          _controller!.play();
                        }
                      }
                      setState(() {});
                    },
                  ),
                  const SizedBox(width: 8),
                  Text(
                    '${_formatDuration(currentPos)} / ${_formatDuration(totalDur)}',
                    style: GoogleFonts.robotoMono(fontSize: 11, color: AppTheme.textMuted),
                  ),
                ],
              ),
              Row(
                children: [
                  // Jump to Impact Button
                  GestureDetector(
                    onTap: _jumpToImpact,
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                      decoration: BoxDecoration(
                        color: Colors.white.withValues(alpha: 0.06),
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(color: AppTheme.borderSubtle),
                      ),
                      child: Row(
                        children: [
                          const Icon(Icons.bolt_rounded, size: 12, color: AppTheme.accentAmber),
                          const SizedBox(width: 4),
                          Text(
                            'Jump to Contact',
                            style: GoogleFonts.outfit(
                              fontSize: 10,
                              fontWeight: FontWeight.w700,
                              color: AppTheme.accentAmber,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(width: 6),
                  // Replay Runup Button
                  GestureDetector(
                    onTap: _replayFromStart,
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                      decoration: BoxDecoration(
                        color: Colors.white.withValues(alpha: 0.06),
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(color: AppTheme.borderSubtle),
                      ),
                      child: Row(
                        children: [
                          const Icon(Icons.replay_rounded, size: 12, color: AppTheme.accentCyan),
                          const SizedBox(width: 4),
                          Text(
                            'Replay',
                            style: GoogleFonts.outfit(
                              fontSize: 10,
                              fontWeight: FontWeight.w700,
                              color: AppTheme.accentCyan,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            ],
          ),
          const SizedBox(height: 6),

          // Bottom Clip Name Footer
          Row(
            children: [
              Expanded(
                child: Text(
                  widget.sampleName ?? 'Penalty Video Analysis',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.inter(fontSize: 11, color: AppTheme.textSubtle),
                ),
              ),
              const SizedBox(width: 8),
              Text(
                'Contact: Frame #${widget.strikeFrameIdx}',
                style: GoogleFonts.robotoMono(fontSize: 10, color: AppTheme.accentEmerald),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
