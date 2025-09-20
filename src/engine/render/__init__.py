"""
どこで: `engine.render` サブパッケージ。
何を: Geometry → GPU 転送・描画の入口。Renderer/LineMesh/Shader を提供。
なぜ: 計算（core/effects）と描画の責務を分離し、GPU リソース管理を局所化するため。
"""
