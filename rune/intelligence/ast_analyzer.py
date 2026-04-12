"""AST analyzer using tree-sitter for multi-language support.

Ported from src/intelligence/ast-analyzer.ts - TS Compiler API (TS-only)
replaced with tree-sitter (40+ languages).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rune.utils.logger import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class Symbol:
    name: str
    kind: str  # function | class | method | variable | import
    start_line: int
    end_line: int
    file_path: str = ""
    parent: str = ""  # enclosing class/module


@dataclass(slots=True)
class FileAnalysis:
    path: str
    language: str
    symbols: list[Symbol] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)


# Language extension to tree-sitter language module mapping
_LANG_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "c_sharp",
    ".swift": "swift",
    ".kt": "kotlin",
    ".lua": "lua",
    ".sh": "bash",
    ".bash": "bash",
}


def is_code_extension(ext: str) -> bool:
    """Return True if the extension maps to a known programming language."""
    return ext.lower() in _LANG_MAP


class ASTAnalyzer:
    """Multi-language AST analysis using tree-sitter."""

    def __init__(self) -> None:
        self._parsers: dict[str, Any] = {}  # language name -> Parser

    def _get_parser(self, language: str) -> Any:
        """Lazy-load a tree-sitter parser for the given language.

        Tries individual tree-sitter-{lang} packages first, then falls back
        to tree-sitter-language-pack (173 languages) if installed.
        """
        if language in self._parsers:
            return self._parsers[language]

        # Try individual package first (e.g., tree-sitter-python)
        try:
            from tree_sitter import Language, Parser

            lang_module = __import__(f"tree_sitter_{language}")
            lang = Language(lang_module.language())
            parser = Parser(lang)
            self._parsers[language] = parser
            return parser
        except (ImportError, AttributeError):
            pass

        # Fallback: tree-sitter-language-pack (optional extra)
        try:
            from tree_sitter_language_pack import get_parser

            parser = get_parser(language)
            self._parsers[language] = parser
            return parser
        except (ImportError, KeyError, Exception) as exc:
            log.debug("tree_sitter_lang_unavailable", language=language, error=str(exc))
            return None

    def detect_language(self, file_path: str | Path) -> str | None:
        """Detect programming language from file extension."""
        ext = Path(file_path).suffix.lower()
        return _LANG_MAP.get(ext)

    def analyze_file(self, file_path: str | Path) -> FileAnalysis | None:
        """Analyze a source file and extract symbols."""
        file_path = Path(file_path)
        language = self.detect_language(file_path)
        if language is None:
            return None

        parser = self._get_parser(language)
        if parser is None:
            return None

        try:
            source = file_path.read_bytes()
            tree = parser.parse(source)
            root = tree.root_node

            symbols: list[Symbol] = []
            imports: list[str] = []

            self._walk_node(root, symbols, imports, str(file_path), language)

            return FileAnalysis(
                path=str(file_path),
                language=language,
                symbols=symbols,
                imports=imports,
            )

        except Exception as exc:
            log.warning("ast_analysis_failed", file=str(file_path), error=str(exc))
            return None

    # Node types that represent function/method definitions across languages
    _FUNC_TYPES = frozenset({
        "function_definition", "function_declaration",     # Python, JS, TS, Go, C
        "method_definition", "method_declaration",         # JS class methods
        "arrow_function",                                  # JS/TS arrow functions
        "function_item",                                   # Rust
        "method",                                          # Ruby
    })

    # Node types that represent class/struct/interface definitions
    _CLASS_TYPES = frozenset({
        "class_definition", "class_declaration",           # Python, JS, TS, Java, Kotlin
        "class_specifier",                                 # C++
        "struct_item",                                     # Rust
        "interface_declaration",                           # TS, Java
        "type_alias_declaration",                          # TS type aliases
        "enum_declaration", "enum_item",                   # Java, Rust
        "namespace_definition",                            # C++
        "module_definition",                               # Ruby
    })

    # Node types for Go type declarations (type Foo struct{})
    _GO_TYPE_TYPES = frozenset({
        "type_spec",                                       # Go: type Foo struct{}
        "type_declaration",                                # Go: type block
    })

    # Import node types
    _IMPORT_TYPES = frozenset({
        "import_statement", "import_from_statement",       # Python
        "import_declaration",                              # Go, Java, TS
        "use_declaration",                                 # Rust
        "require_call",                                    # JS/Ruby require()
    })

    def _walk_node(
        self,
        node: Any,
        symbols: list[Symbol],
        imports: list[str],
        file_path: str,
        language: str,
        parent: str = "",
    ) -> None:
        """Recursively walk the AST to extract symbols."""
        node_type = node.type

        # Function/method definitions
        if node_type in self._FUNC_TYPES:
            name_node = node.child_by_field_name("name")
            if name_node:
                symbols.append(Symbol(
                    name=name_node.text.decode(),
                    kind="method" if parent else "function",
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    file_path=file_path,
                    parent=parent,
                ))

        # Class/struct/interface definitions
        elif node_type in self._CLASS_TYPES:
            name_node = node.child_by_field_name("name")
            if name_node:
                class_name = name_node.text.decode()
                symbols.append(Symbol(
                    name=class_name,
                    kind="class",
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    file_path=file_path,
                    parent=parent,
                ))
                for child in node.children:
                    self._walk_node(child, symbols, imports, file_path, language, class_name)
                return

        # Go type declarations (type Foo struct{})
        elif node_type in self._GO_TYPE_TYPES:
            name_node = node.child_by_field_name("name")
            if name_node:
                symbols.append(Symbol(
                    name=name_node.text.decode(),
                    kind="class",
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    file_path=file_path,
                    parent=parent,
                ))

        # Import statements
        elif node_type in self._IMPORT_TYPES:
            imports.append(node.text.decode())

        # Recurse
        for child in node.children:
            self._walk_node(child, symbols, imports, file_path, language, parent)

    def find_definitions(
        self, file_path: str | Path, name: str,
    ) -> list[Symbol]:
        """Find all definitions of *name* in a file."""
        analysis = self.analyze_file(file_path)
        if analysis is None:
            return []
        return [s for s in analysis.symbols if s.name == name]


# Module singleton
_analyzer: ASTAnalyzer | None = None


def get_ast_analyzer() -> ASTAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = ASTAnalyzer()
    return _analyzer
