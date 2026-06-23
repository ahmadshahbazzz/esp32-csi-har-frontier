import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

// Gate the whole deck behind a password. The unlock page and the unlock API are
// always reachable; static files and Next internals pass through. Everything
// else requires the `deck` cookie set after a correct password.
export function middleware(req: NextRequest) {
  const { pathname } = req.nextUrl;
  if (
    pathname.startsWith("/unlock") ||
    pathname.startsWith("/api/") ||
    /\.(png|jpg|jpeg|svg|ico|pdf|woff2?|css|js)$/.test(pathname)
  ) {
    return NextResponse.next();
  }
  if (req.cookies.get("deck")?.value === "add1ction-ok") {
    return NextResponse.next();
  }
  const url = req.nextUrl.clone();
  url.pathname = "/unlock";
  return NextResponse.redirect(url);
}

export const config = {
  matcher: ["/((?!_next).*)"],
};
