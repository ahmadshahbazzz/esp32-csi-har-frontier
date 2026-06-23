"use client";
import { motion } from "framer-motion";
import GapDrop from "../anim/GapDrop";
import Counter from "../anim/Counter";
import { Eyebrow, Reveal, rise } from "../ui";

export default function GengapSlide() {
  return (
    <div className="slide-pad flex h-full w-full flex-col justify-center">
      <Reveal className="max-w-[1240px]">
        <Eyebrow>finding_02 // the_headline</Eyebrow>
        <motion.h2 variants={rise} className="mt-4 font-display text-3xl font-bold md:text-5xl">
          The real wall is the <span className="alert-text">generalization gap</span>
        </motion.h2>
        <motion.p variants={rise} className="mt-3 max-w-3xl text-lg text-mute">
          Tested on an <span className="text-ink">unseen person</span> (leave-one-user-out), the same
          models collapse versus a random split that lets them memorize each user.
        </motion.p>

        <div className="mt-6 grid grid-cols-1 items-center gap-8 lg:grid-cols-5">
          <motion.div variants={rise} className="lg:col-span-3 panel p-5">
            <div className="font-mono text-xs text-mute">best_model // tiny_cnn · same data, only the split changes</div>
            <GapDrop className="mt-1 h-auto w-full" />
          </motion.div>

          <motion.div variants={rise} className="lg:col-span-2 space-y-4">
            <div className="flex items-end gap-4">
              <div>
                <div className="glow font-display text-5xl font-bold text-trace"><Counter to={96.4} decimals={1} suffix="%" /></div>
                <div className="font-mono text-xs text-mute">random split</div>
              </div>
              <div className="pb-3 text-2xl text-mute">→</div>
              <div>
                <div className="glow-alert font-display text-5xl font-bold text-alert"><Counter to={63.1} decimals={1} suffix="%" /></div>
                <div className="font-mono text-xs text-mute">unseen person</div>
              </div>
            </div>
            <div className="rounded-xl border border-alert/30 bg-alert/10 px-5 py-4 text-center shadow-alert">
              <div className="font-display text-3xl font-bold text-alert">~29 points</div>
              <div className="font-mono text-xs text-mute">average drop across six models</div>
            </div>
            <p className="text-sm text-mute">
              Nothing changes but the split, so the gap is not about model size. A random split lets
              the model memorize each user's body signature; a new person breaks that. The likely fix
              is a short per-user calibration, not a bigger model.
            </p>
          </motion.div>
        </div>
      </Reveal>
    </div>
  );
}
