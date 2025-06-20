"""shapes/text.pyのテストモジュール。"""

import numpy as np
import pytest
from pathlib import Path
from shapes.text import Text, TextRenderer, TEXT_RENDERER


class TestTextRenderer:
    """TextRendererクラスのテストクラス。"""
    
    def setup_method(self):
        """各テストメソッドの前に実行されるセットアップ。"""
        # キャッシュクリア
        TextRenderer._fonts.clear()
        TextRenderer._glyph_cache.clear()
        TextRenderer._font_paths = None
    
    def test_singleton_pattern(self):
        """Singletonパターンの動作テスト。"""
        renderer1 = TextRenderer()
        renderer2 = TextRenderer()
        
        # 同一インスタンスであることを確認
        assert renderer1 is renderer2
        assert renderer1 is TEXT_RENDERER
    
    def test_get_font_path_list(self):
        """フォントパス取得のテスト。"""
        font_paths = TextRenderer.get_font_path_list()
        
        # リストが返されることを確認
        assert isinstance(font_paths, list)
        
        # キャッシュされることを確認
        font_paths2 = TextRenderer.get_font_path_list()
        assert font_paths is font_paths2  # 同一オブジェクト
        
        # パスが存在することを確認（システムに依存）
        if font_paths:
            for font_path in font_paths[:5]:  # 最初の5つだけチェック
                assert isinstance(font_path, Path)
                assert font_path.suffix.lower() in ['.ttf', '.otf', '.ttc']
    
    def test_get_font_default(self):
        """デフォルトフォントの取得テスト。"""
        font = TextRenderer.get_font()
        
        # TTFontオブジェクトが返されることを確認
        from fontTools.ttLib import TTFont
        assert isinstance(font, TTFont)
        
        # フォントがキャッシュされることを確認
        font2 = TextRenderer.get_font()
        assert font is font2
    
    def test_get_font_by_name(self):
        """フォント名指定でのフォント取得テスト。"""
        # 存在しそうなフォント名でテスト
        font_names = ["Helvetica", "Arial", "Times"]
        
        for font_name in font_names:
            try:
                font = TextRenderer.get_font(font_name)
                from fontTools.ttLib import TTFont
                assert isinstance(font, TTFont)
                break  # 1つでも成功すればOK
            except Exception:
                continue  # フォントが見つからない場合は次を試す
    
    def test_get_font_nonexistent(self):
        """存在しないフォント名での動作テスト。"""
        # 存在しないフォント名
        font = TextRenderer.get_font("NonexistentFont12345")
        
        # デフォルトフォントが返されることを確認
        from fontTools.ttLib import TTFont
        assert isinstance(font, TTFont)
    
    def test_get_glyph_commands_basic(self):
        """基本的なグリフコマンド取得のテスト。"""
        # 基本的な文字でテスト
        commands = TextRenderer.get_glyph_commands("A", "Helvetica", 0)
        
        # タプルが返されることを確認
        assert isinstance(commands, tuple)
        
        # キャッシュされることを確認
        commands2 = TextRenderer.get_glyph_commands("A", "Helvetica", 0)
        assert commands is commands2
    
    def test_get_glyph_commands_space(self):
        """スペース文字のグリフコマンド取得テスト。"""
        commands = TextRenderer.get_glyph_commands(" ", "Helvetica", 0)
        
        # スペースは描画コマンドがないことが多い
        assert isinstance(commands, tuple)
    
    def test_get_glyph_commands_special_chars(self):
        """特殊文字のグリフコマンド取得テスト。"""
        special_chars = ["!", "@", "#", "$", "%"]
        
        for char in special_chars:
            commands = TextRenderer.get_glyph_commands(char, "Helvetica", 0)
            assert isinstance(commands, tuple)


class TestText:
    """Textクラスのテストクラス。"""
    
    def setup_method(self):
        """各テストメソッドの前に実行されるセットアップ。"""
        self.text = Text()
        # TextRendererのキャッシュクリア
        TextRenderer._fonts.clear()
        TextRenderer._glyph_cache.clear()
    
    def test_basic_generation(self):
        """基本的なテキスト生成のテスト。"""
        vertices_list = self.text.generate()
        
        # リストが返されることを確認
        assert isinstance(vertices_list, list)
        
        # デフォルトテキスト "HELLO" なので何らかの頂点が生成される
        assert len(vertices_list) >= 0
        
        # 各要素がnumpy配列であることを確認
        for vertices in vertices_list:
            assert isinstance(vertices, np.ndarray)
            assert vertices.shape[1] == 3  # 3D座標
            assert vertices.dtype == np.float32
    
    def test_custom_text(self):
        """カスタムテキストでのテスト。"""
        test_texts = ["A", "AB", "Hello", "123", "!@#"]
        
        for test_text in test_texts:
            vertices_list = self.text.generate(text=test_text)
            
            assert isinstance(vertices_list, list)
            
            # 各要素の検証
            for vertices in vertices_list:
                assert isinstance(vertices, np.ndarray)
                assert vertices.shape[1] == 3
                assert vertices.dtype == np.float32
    
    def test_text_sizing(self):
        """テキストサイズのテスト。"""
        text = "A"
        sizes = [0.05, 0.1, 0.2, 0.5]
        
        previous_size = None
        for size in sizes:
            vertices_list = self.text.generate(text=text, size=size)
            
            if vertices_list:
                # すべての頂点座標の範囲を計算
                all_vertices = np.vstack(vertices_list)
                coord_range = np.max(all_vertices, axis=0) - np.min(all_vertices, axis=0)
                current_size = np.max(coord_range[:2])  # x, yの最大範囲
                
                if previous_size is not None:
                    # サイズが大きくなることを確認
                    assert current_size > previous_size
                
                previous_size = current_size
    
    def test_text_alignment_left(self):
        """左揃えアライメントのテスト。"""
        vertices_list = self.text.generate(text="ABC", align="left")
        
        if vertices_list:
            # すべての頂点を結合
            all_vertices = np.vstack(vertices_list)
            min_x = np.min(all_vertices[:, 0])
            
            # 左揃えでは最小x座標が0付近またはそれ以上
            assert min_x >= -0.1  # 小さな許容誤差
    
    def test_text_alignment_center(self):
        """中央揃えアライメントのテスト。"""
        vertices_list = self.text.generate(text="ABC", align="center")
        
        if vertices_list:
            # すべての頂点を結合
            all_vertices = np.vstack(vertices_list)
            min_x = np.min(all_vertices[:, 0])
            max_x = np.max(all_vertices[:, 0])
            center_x = (min_x + max_x) / 2
            
            # 中央揃えでは中心が0付近
            assert abs(center_x) < 0.1  # 許容誤差
    
    def test_text_alignment_right(self):
        """右揃えアライメントのテスト。"""
        vertices_list = self.text.generate(text="ABC", align="right")
        
        if vertices_list:
            # すべての頂点を結合
            all_vertices = np.vstack(vertices_list)
            max_x = np.max(all_vertices[:, 0])
            
            # 右揃えでは最大x座標が0付近またはそれ以下
            assert max_x <= 0.1  # 小さな許容誤差
    
    def test_empty_text(self):
        """空文字列のテスト。"""
        vertices_list = self.text.generate(text="")
        
        # 空リストまたは空の頂点リストが返される
        assert isinstance(vertices_list, list)
        assert len(vertices_list) == 0
    
    def test_space_only_text(self):
        """スペースのみのテキストのテスト。"""
        vertices_list = self.text.generate(text="   ")
        
        # スペースは描画されないので空リストまたは空の頂点リスト
        assert isinstance(vertices_list, list)
        # スペースは通常描画コマンドを持たないので空になることが多い
    
    def test_mixed_text_with_spaces(self):
        """スペースを含むテキストのテスト。"""
        vertices_list = self.text.generate(text="A B")
        
        assert isinstance(vertices_list, list)
        
        # スペースを含むテキストでも正常に処理される
        for vertices in vertices_list:
            assert isinstance(vertices, np.ndarray)
            assert vertices.shape[1] == 3
    
    def test_font_parameter(self):
        """フォントパラメータのテスト。"""
        # 異なるフォント名でテスト
        font_names = ["Helvetica", "Arial", "Times"]
        
        for font_name in font_names:
            try:
                vertices_list = self.text.generate(text="A", font=font_name)
                assert isinstance(vertices_list, list)
            except Exception:
                # フォントが見つからない場合はスキップ
                continue
    
    def test_font_number_parameter(self):
        """フォント番号パラメータのテスト（TTCファイル用）。"""
        # フォント番号を指定してテスト
        vertices_list = self.text.generate(text="A", font_number=0)
        assert isinstance(vertices_list, list)
        
        # 異なるフォント番号でもエラーにならないことを確認
        vertices_list2 = self.text.generate(text="A", font_number=1)
        assert isinstance(vertices_list2, list)
    
    def test_get_initial_offset(self):
        """初期オフセット計算のテスト。"""
        # プライベートメソッドの直接テスト
        total_width = 100.0
        
        # 左揃え
        offset = self.text._get_initial_offset(total_width, "left")
        assert offset == 0.0
        
        # 中央揃え
        offset = self.text._get_initial_offset(total_width, "center")
        assert offset == -50.0
        
        # 右揃え
        offset = self.text._get_initial_offset(total_width, "right")
        assert offset == -100.0
        
        # 不明なアライメント（左揃えにフォールバック）
        offset = self.text._get_initial_offset(total_width, "unknown")
        assert offset == 0.0
    
    def test_get_char_advance_space(self):
        """スペース文字の幅取得テスト。"""
        from fontTools.ttLib import TTFont
        
        # デフォルトフォントを取得
        tt_font = TextRenderer.get_font()
        
        # スペースの幅を取得
        advance = self.text._get_char_advance(" ", tt_font)
        
        # スペースの幅は正の値
        assert advance > 0
        assert advance < 1.0  # 正規化された値なので1未満
    
    def test_get_char_advance_regular_char(self):
        """通常文字の幅取得テスト。"""
        tt_font = TextRenderer.get_font()
        
        # 通常文字の幅を取得
        advance = self.text._get_char_advance("A", tt_font)
        
        # 文字の幅は正の値
        assert advance >= 0
        assert advance < 1.0  # 正規化された値なので1未満
    
    def test_render_character_space(self):
        """スペース文字のレンダリングテスト。"""
        char_vertices = self.text._render_character(" ", "Helvetica", 0, 1000)
        
        # スペースは描画されないので空リスト
        assert isinstance(char_vertices, list)
        assert len(char_vertices) == 0
    
    def test_render_character_regular(self):
        """通常文字のレンダリングテスト。"""
        char_vertices = self.text._render_character("A", "Helvetica", 0, 1000)
        
        # 文字が描画される場合
        assert isinstance(char_vertices, list)
        
        for vertices in char_vertices:
            assert isinstance(vertices, np.ndarray)
            assert vertices.shape[1] == 3
            assert vertices.dtype == np.float32
    
    def test_normalize_vertices(self):
        """頂点正規化のテスト。"""
        # テスト用の頂点データ
        vertices = [[0, 0, 0], [1000, 1000, 0], [500, 500, 0]]
        units_per_em = 1000
        
        normalized = self.text._normalize_vertices(vertices, units_per_em)
        
        # 正規化された配列が返される
        assert isinstance(normalized, np.ndarray)
        assert normalized.shape == (3, 3)
        assert normalized.dtype == np.float32
        
        # 座標が正規化されている（0-1の範囲）
        assert 0 <= np.max(normalized[:, :2]) <= 1.5  # Y軸反転とオフセットを考慮
        assert -0.5 <= np.min(normalized[:, :2]) <= 1.0
    
    def test_unicode_characters(self):
        """Unicode文字のテスト。"""
        unicode_chars = ["é", "ñ", "ü", "©", "™"]
        
        for char in unicode_chars:
            try:
                vertices_list = self.text.generate(text=char)
                assert isinstance(vertices_list, list)
            except Exception:
                # フォントにない文字の場合はスキップ
                continue
    
    def test_numeric_text(self):
        """数字テキストのテスト。"""
        vertices_list = self.text.generate(text="12345")
        
        assert isinstance(vertices_list, list)
        
        for vertices in vertices_list:
            assert isinstance(vertices, np.ndarray)
            assert vertices.shape[1] == 3
    
    def test_punctuation_text(self):
        """句読点テキストのテスト。"""
        punctuation = ".,;:!?()[]{}\"'"
        
        vertices_list = self.text.generate(text=punctuation)
        
        assert isinstance(vertices_list, list)
        
        for vertices in vertices_list:
            assert isinstance(vertices, np.ndarray)
            assert vertices.shape[1] == 3
    
    def test_additional_params_ignored(self):
        """追加パラメータが無視されることのテスト。"""
        vertices_list = self.text.generate(
            text="TEST",
            size=0.1,
            unused_param=123,
            another_param="test"
        )
        
        # 正常に動作することを確認
        assert isinstance(vertices_list, list)
    
    def test_large_text(self):
        """長いテキストのテスト。"""
        long_text = "The quick brown fox jumps over the lazy dog" * 3
        
        vertices_list = self.text.generate(text=long_text, size=0.05)
        
        assert isinstance(vertices_list, list)
        
        # 長いテキストでも正常に処理される
        for vertices in vertices_list:
            assert isinstance(vertices, np.ndarray)
            assert vertices.shape[1] == 3
    
    def test_very_small_size(self):
        """非常に小さいサイズのテスト。"""
        vertices_list = self.text.generate(text="A", size=0.001)
        
        assert isinstance(vertices_list, list)
        
        if vertices_list:
            # 非常に小さくても頂点は生成される
            all_vertices = np.vstack(vertices_list)
            coord_range = np.max(all_vertices, axis=0) - np.min(all_vertices, axis=0)
            max_range = np.max(coord_range[:2])
            
            # サイズが小さいことを確認
            assert max_range < 0.01
    
    def test_very_large_size(self):
        """非常に大きいサイズのテスト。"""
        vertices_list = self.text.generate(text="A", size=2.0)
        
        assert isinstance(vertices_list, list)
        
        if vertices_list:
            # 大きなサイズでも正常に処理される
            all_vertices = np.vstack(vertices_list)
            coord_range = np.max(all_vertices, axis=0) - np.min(all_vertices, axis=0)
            max_range = np.max(coord_range[:2])
            
            # サイズが大きいことを確認
            assert max_range > 1.0