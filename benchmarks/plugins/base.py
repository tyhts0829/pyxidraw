#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ベンチマークプラグインシステム基底クラス

プラグイン可能なベンチマークシステムのアーキテクチャを提供します。
エラーハンドリング強化版。
"""

import importlib
import inspect
import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from benchmarks.core.types import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkTarget,
    ModuleFeatures,
)
from benchmarks.core.exceptions import ModuleDiscoveryError, BenchmarkError, benchmark_operation

# ロガー設定
logger = logging.getLogger(__name__)


class PluginLoadError(BenchmarkError):
    """プラグイン読み込みエラー"""
    def __init__(self, plugin_name: str, reason: str, original_error: Exception = None):
        self.plugin_name = plugin_name
        self.reason = reason
        self.original_error = original_error
        super().__init__(f"Plugin '{plugin_name}' load failed: {reason}")


class PluginExecutionError(BenchmarkError):
    """プラグイン実行エラー"""
    def __init__(self, plugin_name: str, operation: str, original_error: Exception = None):
        self.plugin_name = plugin_name
        self.operation = operation
        self.original_error = original_error
        super().__init__(f"Plugin '{plugin_name}' execution failed during {operation}")


@contextmanager
def plugin_operation(operation_name: str, plugin_name: str = None):
    """プラグイン操作のコンテキストマネージャー"""
    try:
        logger.debug(f"Starting plugin operation: {operation_name}")
        yield
        logger.debug(f"Completed plugin operation: {operation_name}")
    except ImportError as e:
        logger.error(f"{operation_name} failed - Import error: {e}")
        raise PluginLoadError(plugin_name or "unknown", f"Import failed: {e}", e)
    except AttributeError as e:
        logger.error(f"{operation_name} failed - Attribute error: {e}")
        raise PluginLoadError(plugin_name or "unknown", f"Attribute not found: {e}", e)
    except ModuleDiscoveryError as e:
        logger.error(f"{operation_name} failed - Module discovery error: {e}")
        raise  # 既にカスタム例外なのでそのまま再発生
    except Exception as e:
        logger.warning(f"{operation_name} failed - Unexpected error: {e}")
        raise PluginLoadError(plugin_name or "unknown", f"Unexpected error: {e}", e)


class BenchmarkPlugin(ABC):
    """ベンチマークプラグインの基底クラス"""
    
    def __init__(self, name: str, config: BenchmarkConfig):
        self.name = name
        self.config = config
        self._targets: Optional[List[BenchmarkTarget]] = None
    
    @property
    @abstractmethod
    def plugin_type(self) -> str:
        """プラグインの種類を返す"""
        pass
    
    @abstractmethod
    def discover_targets(self) -> List[BenchmarkTarget]:
        """ベンチマーク対象を発見する"""
        pass
    
    @abstractmethod
    def create_benchmark_target(self, target_name: str, **kwargs) -> BenchmarkTarget:
        """ベンチマーク対象を作成する"""
        pass
    
    @abstractmethod
    def analyze_target_features(self, target: BenchmarkTarget) -> ModuleFeatures:
        """対象の特性を分析する"""
        pass
    
    def get_targets(self, refresh: bool = False) -> List[BenchmarkTarget]:
        """ベンチマーク対象リストを取得（キャッシュあり）"""
        try:
            if self._targets is None or refresh:
                with plugin_operation(f"Discover targets for plugin {self.name}", self.name):
                    targets = self.discover_targets()
                    # 各ターゲットの妥当性を検証
                    valid_targets = []
                    for target in targets:
                        if self.validate_target(target):
                            valid_targets.append(target)
                        else:
                            logger.warning(f"Invalid target discovered: {getattr(target, 'name', 'unknown')}")
                    
                    self._targets = valid_targets
                    logger.info(f"Discovered {len(valid_targets)} valid targets for plugin {self.name}")
            
            return self._targets or []
            
        except PluginLoadError:
            logger.error(f"Failed to get targets for plugin {self.name}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting targets for plugin {self.name}: {e}")
            return []
    
    def is_target_enabled(self, target_name: str) -> bool:
        """対象が有効かどうかをチェック"""
        # 設定ファイルから有効/無効を判定
        return True  # デフォルトは有効
    
    def get_target_config(self, target_name: str) -> Dict[str, Any]:
        """対象固有の設定を取得"""
        return {}
    
    def validate_target(self, target: BenchmarkTarget) -> bool:
        """ベンチマーク対象の妥当性を検証（強化版）"""
        if not target:
            logger.debug("Target validation failed - target is None or empty")
            return False
        
        # 必須属性のチェック
        required_attributes = ['name', 'execute']
        missing_attributes = []
        
        for attr in required_attributes:
            if not hasattr(target, attr):
                missing_attributes.append(attr)
        
        if missing_attributes:
            logger.warning(f"Target validation failed - missing attributes: {missing_attributes}")
            return False
        
        # executeメソッドが呼び出し可能かチェック
        execute_method = getattr(target, 'execute', None)
        if not callable(execute_method):
            logger.warning("Target validation failed - execute method is not callable")
            return False
        
        # 名前が有効な文字列かチェック
        try:
            name = getattr(target, 'name', None)
            if not isinstance(name, str) or not name.strip():
                logger.warning("Target validation failed - invalid name")
                return False
        except Exception as e:
            logger.warning(f"Target validation failed - error accessing name: {e}")
            return False
        
        logger.debug(f"Target validation passed for: {name}")
        return True


class PluginManager:
    """プラグイン管理クラス"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.plugins: Dict[str, BenchmarkPlugin] = {}
        self._auto_discover_plugins()
    
    def register_plugin(self, plugin: BenchmarkPlugin) -> None:
        """プラグインを登録する"""
        self.plugins[plugin.name] = plugin
    
    def unregister_plugin(self, plugin_name: str) -> None:
        """プラグインの登録を解除する"""
        if plugin_name in self.plugins:
            del self.plugins[plugin_name]
    
    def get_plugin(self, plugin_name: str) -> Optional[BenchmarkPlugin]:
        """プラグインを取得する"""
        return self.plugins.get(plugin_name)
    
    def get_plugins_by_type(self, plugin_type: str) -> List[BenchmarkPlugin]:
        """種類別にプラグインを取得する"""
        return [
            plugin for plugin in self.plugins.values()
            if plugin.plugin_type == plugin_type
        ]
    
    def get_all_plugins(self) -> List[BenchmarkPlugin]:
        """すべてのプラグインを取得する"""
        return list(self.plugins.values())
    
    def get_all_targets(self) -> Dict[str, List[BenchmarkTarget]]:
        """すべてのプラグインからベンチマーク対象を取得"""
        all_targets = {}
        successful_plugins = []
        failed_plugins = []
        
        for plugin in self.plugins.values():
            try:
                with plugin_operation(f"Get targets from plugin {plugin.name}", plugin.name):
                    targets = plugin.get_targets()
                    # 各ターゲットにプラグイン名を付与（後段のメタ用）
                    for t in targets:
                        try:
                            setattr(t, 'plugin_name', plugin.name)
                        except Exception:
                            pass
                    all_targets[plugin.name] = targets
                    successful_plugins.append(plugin.name)
                    logger.debug(f"Successfully retrieved {len(targets)} targets from plugin {plugin.name}")
                    
            except PluginExecutionError as e:
                logger.error(f"Plugin execution error getting targets from {plugin.name}: {e}")
                all_targets[plugin.name] = []
                failed_plugins.append((plugin.name, str(e)))
            except Exception as e:
                logger.warning(f"Unexpected error getting targets from plugin {plugin.name}: {e}")
                all_targets[plugin.name] = []
                failed_plugins.append((plugin.name, f"Unexpected error: {e}"))
        
        logger.info(f"Target retrieval completed: {len(successful_plugins)} successful, {len(failed_plugins)} failed")
        
        if failed_plugins:
            logger.warning(f"Failed plugins: {[name for name, _ in failed_plugins]}")
        
        return all_targets
    
    def discover_plugin_classes(self, module_path: str) -> List[Type[BenchmarkPlugin]]:
        """モジュールからプラグインクラスを発見する"""
        plugin_classes = []
        
        try:
            module = importlib.import_module(module_path)
            
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BenchmarkPlugin) and 
                    obj is not BenchmarkPlugin):
                    plugin_classes.append(obj)
        
        except ImportError as e:
            logger.error(f"Failed to import module {module_path}: {e}")
            raise PluginLoadError(module_path, f"Import failed: {e}", e)
        except PluginLoadError:
            raise  # 既にカスタム例外なので再発生
        except Exception as e:
            logger.error(f"Unexpected error discovering plugins from {module_path}: {e}")
            raise PluginLoadError(module_path, f"Unexpected discovery error: {e}", e)
        
        return plugin_classes
    
    def _auto_discover_plugins(self) -> None:
        """プラグインを自動発見する（強化版）"""
        plugin_modules = [
            "benchmarks.plugins.effects",
            "benchmarks.plugins.shapes",
        ]
        
        successful_plugins = []
        failed_plugins = []
        
        logger.info("Starting automatic plugin discovery")
        
        for module_path in plugin_modules:
            try:
                with plugin_operation(f"Auto-discover from {module_path}"):
                    plugin_classes = self.discover_plugin_classes(module_path)
                    
                    for plugin_class in plugin_classes:
                        plugin_name = plugin_class.__name__.replace("BenchmarkPlugin", "").lower()
                        
                        try:
                            with plugin_operation(f"Initialize plugin {plugin_name}", plugin_name):
                                plugin_instance = plugin_class(plugin_name, self.config)
                                self.register_plugin(plugin_instance)
                                successful_plugins.append(plugin_name)
                                logger.debug(f"Successfully registered plugin: {plugin_name}")
                                
                        except PluginLoadError as e:
                            failed_plugins.append((plugin_name, str(e)))
                            logger.error(f"Failed to initialize plugin {plugin_name}: {e}")
                        except Exception as e:
                            failed_plugins.append((plugin_name, f"Unexpected error: {e}"))
                            logger.error(f"Unexpected error initializing plugin {plugin_name}: {e}")
                        
            except PluginLoadError as e:
                failed_plugins.append((module_path, str(e)))
                logger.warning(f"Failed to load plugins from {module_path}: {e}")
            except Exception as e:
                failed_plugins.append((module_path, f"Critical error: {e}"))
                logger.error(f"Critical error during plugin discovery from {module_path}: {e}")
        
        # 結果サマリ
        logger.info(f"Plugin discovery completed: {len(successful_plugins)} successful, {len(failed_plugins)} failed")
        
        if successful_plugins:
            logger.info(f"Successfully loaded plugins: {successful_plugins}")
        
        if failed_plugins:
            logger.warning(f"Failed to load plugins: {[name for name, _ in failed_plugins]}")
            
        # 最低限のプラグインがロードされているかチェック
        if len(successful_plugins) == 0:
            logger.error("No plugins were successfully loaded! This may cause benchmark failures.")
        elif len(successful_plugins) < len(plugin_modules):
            logger.warning(f"Only {len(successful_plugins)}/{len(plugin_modules)} plugin modules loaded successfully")


class BaseBenchmarkTarget:
    """ベンチマーク対象の基底実装"""
    
    def __init__(self, name: str, execute_func: callable, *, tags: Optional[List[str]] = None, **metadata):
        self.name = name
        self._execute_func = execute_func
        self.metadata = metadata
        self.tags: List[str] = list(tags) if tags else []
    
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """ベンチマーク対象を実行"""
        return self._execute_func(*args, **kwargs)
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """メタデータを取得"""
        return self.metadata.get(key, default)
    
    def set_metadata(self, key: str, value: Any) -> None:
        """メタデータを設定"""
        self.metadata[key] = value


class ModuleBenchmarkTarget(BaseBenchmarkTarget):
    """モジュールベースのベンチマーク対象"""
    
    def __init__(self, name: str, module_name: str, function_name: str, **metadata):
        self.module_name = module_name
        self.function_name = function_name
        self._initialized = False
        
        # 動的に関数を取得（詳細エラーハンドリング付き）
        try:
            with plugin_operation(f"Load module {module_name}"):
                module = importlib.import_module(module_name)
                logger.debug(f"Successfully imported module {module_name}")
                
            with plugin_operation(f"Get function {function_name} from {module_name}"):
                if not hasattr(module, function_name):
                    available_functions = [attr for attr in dir(module) if callable(getattr(module, attr, None))]
                    raise AttributeError(f"Function '{function_name}' not found in module '{module_name}'. Available functions: {available_functions[:10]}")
                
                func = getattr(module, function_name)
                
                if not callable(func):
                    raise AttributeError(f"'{function_name}' in module '{module_name}' is not callable")
                
                logger.debug(f"Successfully retrieved function {function_name} from {module_name}")
            
            super().__init__(name, func, **metadata)
            self._initialized = True
            
        except PluginLoadError:
            logger.error(f"Plugin load error initializing ModuleBenchmarkTarget: {name}")
            raise ModuleDiscoveryError(f"Failed to load {function_name} from {module_name}", module_name)
        except Exception as e:
            logger.error(f"Critical error initializing ModuleBenchmarkTarget {name}: {e}")
            raise ModuleDiscoveryError(f"Critical initialization error for {function_name} from {module_name}", module_name)
    
    def reload_function(self) -> None:
        """関数を再読み込み（強化版）"""
        try:
            with plugin_operation(f"Reload module {self.module_name}"):
                module = importlib.import_module(self.module_name)
                importlib.reload(module)
                
            with plugin_operation(f"Re-get function {self.function_name}"):
                self._execute_func = getattr(module, self.function_name)
                
            logger.debug(f"Successfully reloaded function {self.function_name} from {self.module_name}")
            
        except PluginLoadError:
            logger.error(f"Plugin load error reloading {self.function_name} from {self.module_name}")
            raise ModuleDiscoveryError(f"Failed to reload {self.function_name} from {self.module_name}", self.module_name)
        except Exception as e:
            logger.error(f"Unexpected error reloading {self.function_name} from {self.module_name}: {e}")
            raise ModuleDiscoveryError(f"Failed to reload {self.function_name} from {self.module_name}", self.module_name)
    
    def execute(self, *args, **kwargs) -> Any:
        """ベンチマーク対象を実行（安全性チェック付き）"""
        if not self._initialized:
            raise RuntimeError(f"ターゲット {self.name} は正しく初期化されていません")
        
        try:
            return super().execute(*args, **kwargs)
        except Exception as e:
            logger.error(f"Execution failed for target {self.name}: {e}")
            raise


class ParametrizedBenchmarkTarget(BaseBenchmarkTarget):
    """パラメータ化されたベンチマーク対象"""
    
    def __init__(self, name: str, base_func: callable, parameters: Dict[str, Any], *, tags: Optional[List[str]] = None, **metadata):
        self.base_func = base_func
        self.parameters = parameters
        
        # base_funcを直接使用（シリアライズ可能）
        super().__init__(name, base_func, tags=tags, **metadata)
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """パラメータを取得"""
        return self.parameters.get(key, default)
    
    def set_parameter(self, key: str, value: Any) -> None:
        """パラメータを設定"""
        self.parameters[key] = value
    
    def get_all_parameters(self) -> Dict[str, Any]:
        """すべてのパラメータを取得"""
        return self.parameters.copy()


# 便利関数
def create_plugin_manager(config: BenchmarkConfig) -> PluginManager:
    """プラグインマネージャーを作成する便利関数"""
    return PluginManager(config)


def discover_all_targets(config: BenchmarkConfig) -> Dict[str, List[BenchmarkTarget]]:
    """すべてのプラグインからベンチマーク対象を発見する便利関数"""
    manager = create_plugin_manager(config)
    return manager.get_all_targets()
