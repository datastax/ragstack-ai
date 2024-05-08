'use strict'

const { execSync } = require('child_process')

module.exports.register = (context) => {
  context.once('sitePublished', () => {
    const logger = context.getLogger('tailwind-processor-extension')
    logger.info('Building Tailwind')
    execSync('npm run tailwindcss', { stdio: 'inherit' })
    logger.info('Tailwind Build Successful')
  })
}
