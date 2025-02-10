"use strict";

const { execSync } = require("child_process");

module.exports.register = (context) => {
  context.once("sitePublished", ({ playbook }) => {
    const logger = context.getLogger('tailwind-processor-extension')
    const outputDir = playbook?.output?.dir || "build/site";
    logger.info("Building Tailwind");
    var configPath = execSync(`find ${outputDir} -name tailwind.config.js`)
      .toString()
      .trim();
    var cssPath = execSync(`find ${outputDir} -name site*.css`)
      .toString()
      .trim();
    logger.info(
      `npm run tailwindcss --tailwind-config-path=${configPath} --css-path=${cssPath}`
    );
    execSync(
      `npm run tailwindcss --tailwind-config-path=${configPath} --css-path=${cssPath}`,
      { stdio: "inherit" }
    );
    logger.info("Tailwind Build Successful");
  });
};
