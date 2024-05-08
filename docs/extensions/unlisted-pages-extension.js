module.exports.register = function ({ config }) {
  const { addToNavigation, unlistedPagesHeading = 'Unlisted Pages' } = config
  const logger = this.getLogger('unlisted-pages-extension')
  this
    .on('navigationBuilt', ({ contentCatalog }) => {
      contentCatalog.getComponents().forEach(({ versions }) => {
        versions.forEach(({ name: component, version, navigation: nav, url: defaultUrl }) => {
          const navEntriesByUrl = getNavEntriesByUrl(nav)
          const unlistedPages = contentCatalog
            .findBy({ component, version, family: 'page' })
            .filter((page) => page.out)
            .reduce((collector, page) => {
              // Check if the 'unlisted-page' attribute is set to true
              if (page.asciidoc.attributes['unlisted-page'] === 'true') {
                  return collector; // Skip this page
              }
              if ((page.pub.url in navEntriesByUrl) || page.pub.url === defaultUrl) return collector
              logger.warn({ file: page.src, source: page.src.origin }, 'detected unlisted page')
              return collector.concat(page)
            }, [])
          if (unlistedPages.length && addToNavigation) {
            nav.push({
              content: unlistedPagesHeading,
              items: unlistedPages.map((page) => {
                return { content: page.asciidoc.navtitle, url: page.pub.url, urlType: 'internal' }
              }),
              root: true,
            })
          }
        })
      })
    })
}

function getNavEntriesByUrl (items = [], accum = {}) {
  items.forEach((item) => {
    if (item.urlType === 'internal') accum[item.url.split('#')[0]] = item
    getNavEntriesByUrl(item.items, accum)
  })
  return accum
}
