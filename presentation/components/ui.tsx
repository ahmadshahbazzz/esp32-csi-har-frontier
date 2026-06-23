"use client";
import { motion } from "framer-motion";
import { container, rise } from "./anim/variants";

/** Mono console eyebrow, e.g. "> finding_01". */
export function Eyebrow({ children }: { children: React.ReactNode }) {
  return (
    <motion.div variants={rise} className="eyebrow flex items-center gap-2">
      <span className="text-amber">{">"}</span>
      <span className="uppercase">{children}</span>
      <motion.span
        className="inline-block h-3 w-[7px] bg-trace"
        animate={{ opacity: [1, 0.15, 1] }}
        transition={{ duration: 1.1, repeat: Infinity }}
      />
    </motion.div>
  );
}

/** Staggered reveal wrapper used by every slide body. */
export function Reveal({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <motion.div variants={container} initial="hidden" animate="show" className={className}>
      {children}
    </motion.div>
  );
}

export { rise };
