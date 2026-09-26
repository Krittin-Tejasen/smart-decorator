/// One "what is the backend doing right now" update from a generation job.
///
/// [stage] is the pipeline stage id reported by the backend (`compose`,
/// `generate`, `critic`, `retry`, `match`, `segment`); [message] is the
/// human-readable status to show as-is; [progress] is overall 0..1.
class GenerationProgress {
  final String stage;
  final String message;
  final double progress;

  const GenerationProgress({
    required this.stage,
    required this.message,
    required this.progress,
  });
}
