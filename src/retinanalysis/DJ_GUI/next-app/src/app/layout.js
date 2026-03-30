export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body style={{ margin: 0}}>
        {/* Layout UI */}
        <main>{children}</main>
      </body>
    </html>
  )
}