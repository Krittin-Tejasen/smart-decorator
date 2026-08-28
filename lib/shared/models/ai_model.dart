/// One entry in the AI-model picker. `providerValue` is what actually gets
/// sent to the backend's `provider` form field on /generate-room - it maps
/// straight onto the values generation.py's AI_IMAGE_PROVIDER already
/// understands ("mock" / "gemini" / "replicate"), so no backend-side
/// mapping is needed. "FLUX.2 klein" is really Replicate under the hood
/// (see backend/.env REPLICATE_MODEL) - the picker just gives it a
/// friendlier name than the raw provider string.
class AiModel {
  final String providerValue;
  final String title;
  final String subtitle;

  const AiModel({
    required this.providerValue,
    required this.title,
    required this.subtitle,
  });

  static const mock = AiModel(
    providerValue: 'mock',
    title: 'Free Preview',
    subtitle: 'Watermarked preview only, no AI call - free',
  );

  static const gemini = AiModel(
    providerValue: 'gemini',
    title: 'Gemini 3.1 Flash Image',
    subtitle: 'Highest quality, real AI redesign - ~2.3 THB/image',
  );

  static const flux = AiModel(
    providerValue: 'replicate',
    title: 'FLUX.2 klein',
    subtitle: 'Faster & cheaper open-source model - ~1 THB/image',
  );

  static const all = [mock, gemini, flux];
}
