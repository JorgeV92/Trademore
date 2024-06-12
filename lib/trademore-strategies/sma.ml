(* Module for Simple Moving Average Calculation *)
module SMA : sig
  (* Type to encapsulate SMA state *)
  type t = {
    period : int;
    prices : float list;
  }

  (* Create a new SMA calculator with a specified period *)
  val create : int -> t

  (* Add a new price to the SMA calculator *)
  val add_price : float -> t -> t

  (* Calculate the current average *)
  val calculate_average : t -> float option
end = struct
  type t = {
    period : int;
    prices : float list;
  }

  let create period = {
    period = max 1 period;  (* Ensure the period is at least 1 *)
    prices = [];
  }

  let add_price price sma =
    let new_prices = price :: sma.prices in
    let trimmed_prices = List.take new_prices sma.period in
    { sma with prices = trimmed_prices }

  let calculate_average sma =
    match sma.prices with
    | [] -> None
    | prices ->
        let sum = List.fold_left (+.) 0.0 prices in
        Some (sum /. float (List.length prices))
end

