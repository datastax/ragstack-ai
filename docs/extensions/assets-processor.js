"use strict";

module.exports.register = (context) => {
  const logger = context.getLogger("assets-processor-extension");

  context.once("uiLoaded", ({ uiCatalog }) => {
    const manifestContents = uiCatalog
      .findByType("asset")
      .find((file) => file.stem === "assets-manifest")
      .contents?.toString();
    if (!manifestContents) {
      logger.error("Could not find assets-manifest.json in the UI bundle.");
      return;
    }
    const manifest = JSON.parse(manifestContents);
    // Add manifest to node global context so it can be accessed by the handlebars helper during createPageComposer
    global.assetsManifest = manifest;
  });

  context.once("pagesComposed", () => {
     // Clean up the global context
     delete global.assetsManifest;
  });
};
