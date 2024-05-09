const logger = require("@antora/logger")("asciidoctor:svg-macro");

/**
 * @example Inline Embedded SVG
 * svg:ROOT:ui/icons/vector.svg[]
 */
function inlineSvgMacro({ contentCatalog, file }) {
  return function () {
    this.process((parent, target, attrs) => {
      svgContent = getSvgContent(target, file, contentCatalog);
      if (!svgContent) return;
      return this.createInlinePass(
        parent,
        svgContent.replace("<svg", `<svg ${htmlAttrs(attrs)}`)
      );
    });
  };
}

/**
 * @example Block Embedded SVG
 * svg::home:diagrams/graphic.svg[alt="My Graphic"]
 */
function blockSvgMacro({ contentCatalog, file }) {
  return function () {
    this.process((parent, target, attrs) => {
      svgContent = getSvgContent(target, file, contentCatalog);
      if (!svgContent) return;
      const svgHtmlAttrs = htmlAttrs({ ...attrs, role: undefined });
      const containerHtmlAttrs = attrs.role
        ? `class="imageblock ${attrs.role}"`
        : 'class="imageblock"';
      const html =
        `<div ${containerHtmlAttrs}><div class="content">` +
        svgContent.replace("<svg", `<svg ${svgHtmlAttrs}`) +
        "</div></div>";
      return this.createBlock(parent, "pass", html);
    });
  };
}

/**
 * This macro relies on the material-icons font being loaded in UI bundle.
 *
 * @example Material Icon
 * icon:material-icons:menu_open[]
 *
 * @example Embedded SVG
 * icon:ROOT:ui/icons/vector.svg[]
 */
function inlineIconMacro({ contentCatalog, file }) {
  return function () {
    this.process((parent, target, attrs) => {
      if (target.startsWith("material-icons")) {
        iconTarget = target
          .replace("material-icons:", "")
          .trim()
          .replace("-", "_");
        return this.createInlinePass(
          parent,
          `<i ${htmlAttrs(attrs, "material-icons icon")}>${iconTarget}</i>`
        );
      } else {
        svgContent = getSvgContent(target, file, contentCatalog);
        if (!svgContent) return;
        return this.createInlinePass(
          parent,
          svgContent.replace("<svg", `<svg ${htmlAttrs(attrs, "svg icon")}`)
        );
      }
    });
  };
}

function getSvgContent(target, file, contentCatalog) {
  svgFile = contentCatalog.resolveResource(target, file.src, "image", [
    "image",
  ]);
  if (!svgFile)
    return logger.error({ target, file }, `target of svg not found: ${target}`);
  svgContent = svgFile.contents.toString();
  if (!svgContent.startsWith("<svg"))
    return logger.error({ target, file }, "file contents must be a valid svg");
  return svgContent;
}

function htmlAttrs({ width, height, role, alt, title }, klass = "svg") {
  return [
    width && `width="${width}"`,
    height && `height="${height}"`,
    role ? `class="${klass} ${role}"` : `class="${klass}"`,
    alt && `aria-label="${alt}"`,
    title && `aria-label="${title}"`,
    (alt || title) && 'role="img"',
  ]
    .filter(Boolean)
    .join(" ");
}

/**
 * @param { import("@asciidoctor/core/types").Asciidoctor.Extensions.Registry } registry
 * @param context
 */
function register(registry, context) {
  registry.inlineMacro("svg", inlineSvgMacro(context));
  registry.blockMacro("svg", blockSvgMacro(context));
  registry.inlineMacro("icon", inlineIconMacro(context));
}

module.exports.register = register;
