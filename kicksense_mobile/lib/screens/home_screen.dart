import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../models/prediction_result.dart';
import '../services/api_service.dart';
import '../theme/app_theme.dart';
import 'analyze_screen.dart';
import 'results_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  List<Map<String, dynamic>> _samples = List.from(ApiService.defaultSamples);
  bool _loadingSamples = false;
  bool _isPredictingSample = false;

  @override
  void initState() {
    super.initState();
    _fetchSamples();
  }

  Future<void> _fetchSamples() async {
    final list = await ApiService.getSampleClips();
    if (mounted && list.isNotEmpty) {
      setState(() {
        _samples = list;
        _loadingSamples = false;
      });
    }
  }

  Future<void> _runSampleKick(String sampleId) async {
    if (_isPredictingSample) return;

    setState(() => _isPredictingSample = true);

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Row(
          children: [
            const SizedBox(
              width: 18,
              height: 18,
              child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
            ),
            const SizedBox(width: 12),
            Text('Analyzing sample kick ($sampleId)...'),
          ],
        ),
        duration: const Duration(seconds: 10),
      ),
    );

    try {
      final PredictionResult result = await ApiService.predictSample(sampleId);
      if (!mounted) return;
      ScaffoldMessenger.of(context).hideCurrentSnackBar();

      final sample = _samples.firstWhere(
        (s) => s['id'] == sampleId,
        orElse: () => <String, dynamic>{},
      );
      final baseUrl = await ApiService.getBaseUrl();
      final videoUrl = '$baseUrl/api/sample-video/$sampleId';
      final sampleName = sample['name'] as String?;
      final trueDir = sample['true_direction'] as String?;
      final truePwr = (sample['true_power'] as num?)?.toDouble();

      if (!mounted) return;
      Navigator.of(context).push(
        MaterialPageRoute(
          builder: (_) => ResultsScreen(
            result: result,
            videoUrl: videoUrl,
            sampleName: sampleName,
            trueDirection: trueDir,
            truePower: truePwr,
          ),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).hideCurrentSnackBar();
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Sample test failed: $e')),
      );
    } finally {
      if (mounted) setState(() => _isPredictingSample = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Top Brand Header
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Row(
                    children: [
                      Container(
                        width: 44,
                        height: 44,
                        decoration: BoxDecoration(
                          gradient: const LinearGradient(
                            colors: [AppTheme.accentEmerald, AppTheme.accentCyan],
                          ),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: const Icon(Icons.sports_soccer_rounded, color: Colors.white, size: 28),
                      ),
                      const SizedBox(width: 12),
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              Text(
                                'KickSense',
                                style: GoogleFonts.outfit(
                                  fontSize: 22,
                                  fontWeight: FontWeight.w800,
                                  color: Colors.white,
                                ),
                              ),
                              const SizedBox(width: 6),
                              Container(
                                padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                                decoration: BoxDecoration(
                                  color: AppTheme.accentEmerald.withValues(alpha: 0.2),
                                  borderRadius: BorderRadius.circular(6),
                                  border: Border.all(color: AppTheme.accentEmerald.withValues(alpha: 0.4)),
                                ),
                                child: Text(
                                  'AI',
                                  style: GoogleFonts.robotoMono(
                                    fontSize: 10,
                                    fontWeight: FontWeight.w700,
                                    color: AppTheme.accentEmerald,
                                  ),
                                ),
                              ),
                            ],
                          ),
                          Text(
                            'Penalty Anticipation & Biomechanics',
                            style: GoogleFonts.inter(
                              fontSize: 11,
                              color: AppTheme.textMuted,
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                  IconButton(
                    icon: const Icon(Icons.info_outline_rounded, color: AppTheme.textMuted),
                    onPressed: () => _showAboutDialog(context),
                  ),
                ],
              ),
              const SizedBox(height: 24),

              // Hero Project Stats Card
              Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  gradient: const LinearGradient(
                    colors: [Color(0xFF0F1E38), Color(0xFF13274F)],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(color: AppTheme.borderBright),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text(
                          'FINAL YEAR PROJECT MODEL',
                          style: GoogleFonts.outfit(
                            fontSize: 11,
                            fontWeight: FontWeight.w700,
                            letterSpacing: 1.2,
                            color: AppTheme.accentCyan,
                          ),
                        ),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                          decoration: BoxDecoration(
                            color: AppTheme.accentEmerald.withValues(alpha: 0.15),
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: Text(
                            'Verified Split',
                            style: GoogleFonts.robotoMono(
                              fontSize: 10,
                              color: AppTheme.accentEmerald,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 14),
                    Row(
                      children: [
                        _buildStat('53.1%', 'Test Acc', AppTheme.accentEmerald),
                        _buildStatDivider(),
                        _buildStat('78.6%', 'Center Prec', AppTheme.accentCyan),
                        _buildStatDivider(),
                        _buildStat('2.12', 'Power MAE', AppTheme.accentAmber),
                        _buildStatDivider(),
                        _buildStat('460', 'Clips', Colors.white),
                      ],
                    ),
                    const SizedBox(height: 14),
                    Text(
                      'Multi-Scale Temporal Inception + BiLSTM network with shift-invariant dual global pooling.',
                      style: GoogleFonts.inter(fontSize: 12, color: AppTheme.textMuted, height: 1.35),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 24),

              // Instant Sample Kicks Section
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    'INSTANT DEMO SAMPLES',
                    style: GoogleFonts.outfit(
                      fontSize: 12,
                      fontWeight: FontWeight.w700,
                      letterSpacing: 1.0,
                      color: AppTheme.textSubtle,
                    ),
                  ),
                  IconButton(
                    iconSize: 18,
                    padding: EdgeInsets.zero,
                    constraints: const BoxConstraints(),
                    icon: const Icon(Icons.refresh_rounded, color: AppTheme.textMuted),
                    onPressed: _fetchSamples,
                  ),
                ],
              ),
              const SizedBox(height: 10),
              if (_loadingSamples && _samples.isEmpty)
                const Center(child: Padding(padding: EdgeInsets.all(12), child: CircularProgressIndicator(strokeWidth: 2)))
              else if (_samples.isEmpty)
                Container(
                  padding: const EdgeInsets.all(14),
                  decoration: BoxDecoration(
                    color: AppTheme.bgSurface,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    'No sample clips found on server. Start backend to test live.',
                    style: GoogleFonts.inter(fontSize: 12, color: AppTheme.textMuted),
                  ),
                )
              else
                SizedBox(
                  height: 105,
                  child: ListView.separated(
                    scrollDirection: Axis.horizontal,
                    itemCount: _samples.length,
                    separatorBuilder: (_, _) => const SizedBox(width: 12),
                    itemBuilder: (context, index) {
                      final s = _samples[index];
                      return InkWell(
                        onTap: () => _runSampleKick(s['id'] as String),
                        borderRadius: BorderRadius.circular(14),
                        child: Container(
                          width: 170,
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            color: AppTheme.bgSurface,
                            borderRadius: BorderRadius.circular(14),
                            border: Border.all(color: AppTheme.borderSubtle),
                          ),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            children: [
                              Text(
                                s['name'] as String? ?? 'Sample',
                                maxLines: 2,
                                overflow: TextOverflow.ellipsis,
                                style: GoogleFonts.outfit(
                                  fontSize: 13,
                                  fontWeight: FontWeight.w600,
                                  color: Colors.white,
                                ),
                              ),
                              Row(
                                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                                children: [
                                  Row(
                                    children: [
                                      const Icon(Icons.bolt_rounded, size: 14, color: AppTheme.accentEmerald),
                                      const SizedBox(width: 4),
                                      Text(
                                        'Tap to Test',
                                        style: GoogleFonts.outfit(
                                          fontSize: 11,
                                          fontWeight: FontWeight.w700,
                                          color: AppTheme.accentEmerald,
                                        ),
                                      ),
                                    ],
                                  ),
                                  const Icon(Icons.play_circle_fill_rounded, size: 20, color: AppTheme.accentCyan),
                                ],
                              ),
                            ],
                          ),
                        ),
                      );
                    },
                  ),
                ),
              const SizedBox(height: 28),

              // How It Works
              Text(
                'SYSTEM PIPELINE',
                style: GoogleFonts.outfit(
                  fontSize: 12,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 1.0,
                  color: AppTheme.textSubtle,
                ),
              ),
              const SizedBox(height: 12),
              _buildStepTile(
                '1',
                'Upload Run-Up Video',
                'Upload penalty video from broadcast or phone camera recording.',
                Icons.videocam_rounded,
              ),
              const SizedBox(height: 8),
              _buildStepTile(
                '2',
                'Smart Contact Detection',
                'MediaPipe tracks 33 3D body keypoints and cuts at strike impact moment.',
                Icons.sports_soccer_rounded,
              ),
              const SizedBox(height: 8),
              _buildStepTile(
                '3',
                'Deep Learning Inference',
                'Multi-scale temporal inception predicts direction, power & posture kinematics.',
                Icons.auto_graph_rounded,
              ),
              const SizedBox(height: 32),

              // Primary Action Button
              ElevatedButton.icon(
                onPressed: () {
                  Navigator.of(context).push(
                    MaterialPageRoute(builder: (_) => const AnalyzeScreen()),
                  );
                },
                icon: const Icon(Icons.add_photo_alternate_rounded),
                label: const Text('ANALYZE PENALTY KICK'),
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 18),
                  backgroundColor: AppTheme.accentEmerald,
                ),
              ),
              const SizedBox(height: 20),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildStat(String val, String label, Color color) {
    return Expanded(
      child: Column(
        children: [
          Text(
            val,
            style: GoogleFonts.outfit(
              fontSize: 18,
              fontWeight: FontWeight.w800,
              color: color,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            label,
            style: GoogleFonts.inter(
              fontSize: 10,
              color: AppTheme.textSubtle,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStatDivider() {
    return Container(
      width: 1,
      height: 26,
      color: Colors.white12,
    );
  }

  Widget _buildStepTile(String step, String title, String subtitle, IconData icon) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppTheme.bgSurface,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppTheme.borderSubtle),
      ),
      child: Row(
        children: [
          Container(
            width: 36,
            height: 36,
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.05),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Icon(icon, size: 18, color: AppTheme.accentCyan),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: GoogleFonts.outfit(
                    fontSize: 14,
                    fontWeight: FontWeight.w700,
                    color: Colors.white,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  subtitle,
                  style: GoogleFonts.inter(
                    fontSize: 11,
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

  void _showAboutDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: AppTheme.bgCard,
        title: Text(
          'About KickSense',
          style: GoogleFonts.outfit(fontWeight: FontWeight.w800),
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Final Year Project\nBahria University Islamabad',
              style: GoogleFonts.outfit(fontSize: 14, fontWeight: FontWeight.w600, color: AppTheme.accentCyan),
            ),
            const SizedBox(height: 8),
            Text(
              'Authors: Ashir Ijaz & Ibrahim Tariq\nSupervisor: Ms. Zainab Iftikhar',
              style: GoogleFonts.inter(fontSize: 13, color: AppTheme.textMuted),
            ),
            const SizedBox(height: 12),
            Text(
              'KickSense anticipates football penalty trajectory (Left, Center, Right) and shot power (1-10) strictly up to foot-ball contact moment using MediaPipe pose estimation and CNN-BiLSTM.',
              style: GoogleFonts.inter(fontSize: 12, color: AppTheme.textSubtle, height: 1.4),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(ctx).pop(),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }
}
