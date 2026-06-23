import TitleSlide from "./01-Title";
import WhySlide from "./02-Why";
import CsiHarSlide from "./03-CsiHar";
import DemoSlide from "./Demo";
import GapSlide from "./04-Gap";
import ApproachSlide from "./05-Approach";
import Finding1Slide from "./06-Finding1";
import BarriersSlide from "./07-Barriers";
import GengapSlide from "./08-Gengap";
import ResultsSlide from "./09-Results";
import TakeawaysSlide from "./10-Takeaways";
import ThankYouSlide from "./11-ThankYou";

import type { ComponentType } from "react";

export const slides: { id: string; label: string; Component: ComponentType }[] = [
  { id: "title", label: "Title", Component: TitleSlide },
  { id: "why", label: "Motivation", Component: WhySlide },
  { id: "csi-har", label: "What is CSI HAR", Component: CsiHarSlide },
  { id: "demo", label: "Live demo", Component: DemoSlide },
  { id: "gap", label: "The gap", Component: GapSlide },
  { id: "approach", label: "Approach", Component: ApproachSlide },
  { id: "finding-1", label: "Deployment fits", Component: Finding1Slide },
  { id: "barriers", label: "Real barriers", Component: BarriersSlide },
  { id: "gengap", label: "Generalization gap", Component: GengapSlide },
  { id: "results", label: "Results", Component: ResultsSlide },
  { id: "takeaways", label: "Takeaways", Component: TakeawaysSlide },
  { id: "thank-you", label: "Thank you", Component: ThankYouSlide },
];
