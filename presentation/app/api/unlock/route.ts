import { NextResponse } from "next/server";

const PASSWORD = "add1ction";

export async function POST(req: Request) {
  let password = "";
  try {
    ({ password } = await req.json());
  } catch {
    password = "";
  }
  if (password === PASSWORD) {
    const res = NextResponse.json({ ok: true });
    res.cookies.set("deck", "add1ction-ok", {
      httpOnly: true,
      sameSite: "lax",
      path: "/",
      maxAge: 60 * 60 * 24 * 30,
    });
    return res;
  }
  return NextResponse.json({ ok: false }, { status: 401 });
}
